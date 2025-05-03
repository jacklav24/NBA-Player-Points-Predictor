/**
 * This file is part of NBA Player Predictor.
 * Copyright (C) 2025 John LaVergne
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU General Public License for more details.
 * <https://www.gnu.org/licenses/>.
 */


import React, { useEffect, useState, useMemo } from 'react';
import axios from 'axios';
import CustomComboboxDropdown from '../components/CustomComboboxDropdown';
import ModelMetrics from '../components/ModelMetrics';
import FeatureBar from '../components/FeatureBar';
import teamLabels from '../constants/teamLabels';
import Predictions from '../components/Predictions';
import ModelRunHistory from '../components/ModelRunHistory';
import { useNavigate } from 'react-router-dom';
import { useCallback } from 'react';
import RunControls from '../components/RunControls';

function HomePage() {
  const [players, setPlayers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [opponents, setOpponents] = useState([]);

  const [player, setPlayer] = useState('');
  const [team, setTeam] = useState('');
  const [opponent, setOpponent] = useState('');
  const [location, setLocation] = useState('');

  const [globalResult, setGlobalResult] = useState(null);
  const [individualResult, setIndividualResult] = useState(null);
  const [blendedResult, setBlendedResult] = useState(null);
  const [insights, setInsights] = useState(null);

  const [loading, setLoading] = useState(false);
  const [reTuning, setReTuning] = useState(false);
  const [reTraining, setReTraining] = useState(false);
  const [saveRun, setSaveRun] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const[playerPredicted, setPlayerPredicted] = useState(false);

  const navigate = useNavigate();

  const filters = useMemo(() => ({ team, player, opponent, location }), [team, player, opponent, location]);
  const onFilterChange = useMemo(() => ({ setTeam, setPlayer, setOpponent, setLocation }), []);


 


  useEffect(() => {
    axios.get('http://localhost:8000/teams').then(res => setTeams(res.data));
    axios.get('http://localhost:8000/opponents').then(res => setOpponents(res.data));
    axios.get('http://localhost:8000/model_insights')
      .then(res => setInsights(res.data))
      .catch(console.error);
    
    }, []);

  useEffect(() => {
    if (team) {
      axios.get(`http://localhost:8000/${team}/players`)
        .then(res => setPlayers(res.data))
        .catch(err => {
          console.error("Error fetching players for team", team, err);
          setPlayers([]);
        });
    } else {
      setPlayers([]);
    }
  }, [team]);


  const fetchInsights = () => {
    axios.get("http://localhost:8000/model_insights")
    .then(res => {setInsights(res.data);
  })
         .catch(console.error);
  };

  const handlePredict = useCallback(async () => { 
    
      if (!player || !team || !opponent) {
        alert("Please select team, player, and opponent.");
        return;
      }
      try {
        setPlayerPredicted(false);
        setLoading(true);
        setGlobalResult(null);
        setIndividualResult(null);
        setBlendedResult(null); 
        setIsPredicting(true);
        console.log("about to fetch")
        const [blendedRes] = await Promise.all([
          axios.post('http://localhost:8000/predict_both', {
            player_name: player,
            team: team,
            opponent: opponent,
            home: location,
            save_run: saveRun,
          }),
        ]);
        console.log("fetched")
        setGlobalResult(blendedRes.data.global_model);
        setIndividualResult(blendedRes.data.individual_model);
        setBlendedResult(blendedRes.data.blended_model)
        setPlayerPredicted(true);

        // re-fetch model insights
        await fetchInsights();

      } catch (err) {
        console.error("Predict error response:", err.response?.data);
        alert("Prediction failed: " + JSON.stringify(err.response?.data));
      }finally {
        setLoading(false);
        setIsPredicting(false);
        
      }
    }, [team, player, opponent, location, saveRun, fetchInsights]);
    const controls = useMemo(() => ({
      kickOffOptimize: async () => {
        setReTuning(true);
        try {
          const res = await fetch("http://localhost:8000/optimize_sync?n_trials=50", { method: 'POST' });
          const data = await res.json();
          console.log("Tuning complete:", data);
          await fetchInsights();    
        } catch(err) {
          console.error(err);
        } finally {
          setReTuning(false);
        }
      },
      kickOffReTrain: async () => {
        setReTraining(true);
        try {
          const res = await fetch("http://localhost:8000/train_global", { method: 'POST' });
          const data = await res.json();
          console.log("Training complete:", data);
          await fetchInsights();    
        } catch(err) {
          console.error(err);
        } finally {
          setReTraining(false);
        }
      },
    }), [fetchInsights]);
      
  

  const status = useMemo(() => ({ loading, reTuning, reTraining }), [loading, reTuning, reTraining]);


  
  
 

    return (
      <div className="min-h-screen bg-[#1e2147] text-[#f5f5f5] p-8">
        <h1 className="text-3xl font-bold mb-8 text-center">
          NBA Player Points Predictor
        </h1>
        <button
        onClick={() => navigate('/runs')}
        className="absolute top-6 right-6 bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded-lg text-sm"
      >
        Run History
      </button>
  
        <RunControls
        teams={teams}
        players={players}
        opponents={opponents}
        teamLabels={teamLabels}
        handlePredict={handlePredict}
        filters={filters}
        onFilterChange={onFilterChange}
        controls={controls}
        status={status}
        saveRun={saveRun}
        onToggleSaveRun={() => setSaveRun(v => !v)}
      />
  
        <Predictions indiv={individualResult} global={globalResult} blended={blendedResult} playerName={player} />
         
  
        {insights && (
              <ModelMetrics
                metrics={insights.metrics}
                featureImportance={insights.feature_importance}
                playerPredicted={playerPredicted}
              />
        )}
      </div>
    );
  }




export default HomePage;