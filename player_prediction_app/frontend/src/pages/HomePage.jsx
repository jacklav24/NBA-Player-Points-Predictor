
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import CustomComboboxDropdown from '../components/CustomComboboxDropdown';
import ModelMetrics from '../components/ModelMetrics';
import FeatureBar from '../components/FeatureBar';
import teamLabels from '../constants/teamLabels';
import Predictions from '../components/Predictions';
import ModelRunHistory from '../components/ModelRunHistory';
import { useNavigate } from 'react-router-dom';

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
      setBlendedResult(null); 
      setIsPredicting(true);

      const [blendedRes] = await Promise.all([
        axios.post('http://localhost:8000/predict_both', {
          player_name: player,
          team: team,
          opponent: opponent,
          home: location,
          save_run: saveRun,
        }),
      ]);

      setGlobalResult(blendedRes.data.global_model);
      setIndividualResult(blendedRes.data.individual_model);
      setBlendedResult(blendedRes.data.blended_model)
      setPlayerPredicted(true);

      // re-fetch model insights
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

  const kickOffReTrain = async () => {
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
  };

 

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
  
        <div className="w-full max-w-7xl mx-auto bg-[#2a2d55] p-6 rounded-xl shadow-lg">
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
            <button className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px]"
            onClick={kickOffReTrain}
            disabled={reTraining}>
            {reTraining ? 'Training...' : 'Re-Train Global Model'}</button>
            <button
              onClick={() => setSaveRun(v => !v)}
              className={`
                px-3 py-2 rounded-lg h-[42px] transition 
                ${saveRun
                  ? 'bg-indigo-500 hover:bg-indigo-400 shadow-inner scale-95 text-white'
                  : 'bg-gray-700 hover:bg-gray-600 text-white'}`}>            
                      Save Run: {saveRun ? "On" : "Off"}
            </button>
          </div>
          {(loading || reTuning) && (
            <div className="text-center mt-2 animate-pulse">
              <span className="text-indigo-300">Calculating results...</span>
            </div>
          )}
        </div>
  
        <Predictions indiv={individualResult} global={globalResult} blended={blendedResult} playerName={player} />
         
  
        {insights && (
          <div className="max-w-7xl mx-auto mt-12 p-6 bg-[#2a2d55] shadow-xl">
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




export default HomePage;