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

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import CustomComboboxDropdown from './components/CustomComboboxDropdown';
import ModelMetrics from './components/ModelMetrics';
import FeatureBar from './components/FeatureBar';
import teamLabels from './constants/teamLabels';
import Predictions from './components/Predictions';

function App() {
  const [players, setPlayers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [opponents, setOpponents] = useState([]);

  const [player, setPlayer] = useState('');
  const [team, setTeam] = useState('');
  const [opponent, setOpponent] = useState('');
  const [location, setLocation] = useState('');

  const [globalResult, setGlobalResult] = useState(null);
  const [individualResult, setIndividualResult] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reTuning, setReTuning] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const[playerPredicted, setPlayerPredicted] = useState(false);

 


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

  const handlePredict = async () => {
    if (!player || !team || !opponent) {
      alert("Please select team, player, and opponent.");
      return;
    }
    try {
      setPlayerPredicted(false);
      setLoading(true);
      setGlobalResult(null);
      setIndividualResult(null);
      setIsPredicting(true);
      const [globalRes, individualRes] = await Promise.all([
        axios.post('http://localhost:8000/global_predict', {
          player_name: player,
          team: team,
          opponent: opponent,
          home: location,
        }),
        axios.post('http://localhost:8000/predict', {
          player_name: player,
          team: team,
          opponent: opponent,
          home: location,
        })
      ]);

      setGlobalResult(globalRes.data);
      setIndividualResult(individualRes.data);
      setPlayerPredicted(true);

      // re fetch model insights
      await fetchInsights();

    } catch (err) {
      console.error("Predict error:", err.response?.data || err);
      alert("Prediction failed; check console for details.");
    }finally {
      setLoading(false);
      setIsPredicting(false);
    }
  };


  
  const fetchInsights = () => {
    axios.get("http://localhost:8000/model_insights")
         .then(res => setInsights(res.data))
         .catch(console.error);
  };

  const kickOffOptimize = async () => {
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
  };

 

    return (
      <div className="min-h-screen bg-[#1e2147] text-[#f5f5f5] p-8">
        <h1 className="text-3xl font-bold mb-8 text-center">
          NBA Player Points Predictor
        </h1>
  
        <div className="max-w-6xl mx-auto bg-[#2a2d55] p-6 rounded-xl shadow-lg">
          <div className="flex flex-wrap md:flex-nowrap justify-between items-end gap-4 mb-4">
            <CustomComboboxDropdown label="Team" options={teams} value={team} onChange={v => { setTeam(v); setPlayer(''); }} displayMap={teamLabels} />
            <CustomComboboxDropdown label="Player" options={players} value={player} onChange={setPlayer} disabled={!team} />
            <CustomComboboxDropdown label="Opponent" options={opponents} value={opponent} onChange={setOpponent} displayMap={teamLabels}/>
            <CustomComboboxDropdown label="Location" options={['Home','Away']} value={location} onChange={setLocation} displayMap={{ Home: 'Home (H)', Away: 'Away (A)' }} />
            <button
              className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px]"
              onClick={handlePredict}
              disabled={loading | reTuning}
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>
            <button className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px]"
                    onClick={kickOffOptimize}
                    disabled={reTuning}>
                    {reTuning ? 'Tuning...' : 'Re-Tune Hyperparameters'}</button>
          </div>
  
          {loading && (
            <div className="text-center mt-2 animate-pulse">
              <span className="text-indigo-300">Calculating results...</span>
            </div>
          )}
        </div>
  
        <Predictions indiv={individualResult} global={globalResult} playerName={player} />
  
        {insights && (
          <div className="max-w-4xl mx-auto mt-12 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
            <h2 className="text-2xl font-semibold text-center text-indigo-300 mb-4">
              Model Diagnostics
            </h2>
            <ModelMetrics
              metrics={insights.metrics}
              featureImportance={insights.feature_importance}
              playerPredicted={playerPredicted}
            />
          </div>
        )}
      </div>
    );
  }




export default App;