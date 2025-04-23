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
      setLoading(true);
      setGlobalResult(null);
      setIndividualResult(null);

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
    } catch (err) {
      console.error("Predict error:", err.response?.data || err);
      alert("Prediction failed; check console for details.");
    }finally {
      setLoading(false);
    }
  };
  const formatPlayerName = (name) => name
    .split(/[_-]/)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ');

    return (
      <div className="min-h-screen bg-[#1e2147] text-[#f5f5f5] p-8">
        <h1 className="text-3xl font-bold mb-8 text-center">
          NBA Player Points Predictor
        </h1>
  
        <div className="max-w-6xl mx-auto bg-[#2a2d55] p-6 rounded-xl shadow-lg">
          <div className="flex flex-wrap md:flex-nowrap justify-between items-end gap-4 mb-4">
            <CustomComboboxDropdown label="Team" options={teams} value={team} onChange={v => { setTeam(v); setPlayer(''); }} />
            <CustomComboboxDropdown label="Player" options={players} value={player} onChange={setPlayer} disabled={!team} />
            <CustomComboboxDropdown label="Opponent" options={opponents} value={opponent} onChange={setOpponent} />
            <CustomComboboxDropdown label="Location" options={['Home','Away']} value={location} onChange={setLocation} />
            <button
              className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px]"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
  
          {loading && (
            <div className="text-center mt-2 animate-pulse">
              <span className="text-indigo-300">Calculating results...</span>
            </div>
          )}
        </div>
  
        {(globalResult || individualResult) && (
          <div className="max-w-7xl mx-auto mt-8 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
            <h2 className="text-2xl font-semibold mb-4 text-center text-indigo-200">
              Prediction Results
            </h2>
  
            {globalResult && (
              <>
                <h3 className="text-xl font-medium mb-4 text-center">Global Model</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(globalResult).map(([name, d]) =>
                    <ResultBox key={name} title={name} data={d} />
                  )}
                </div>
              </>
            )}
  
            {individualResult && (
              <>
                <h3 className="text-xl font-medium mb-4 text-center">Individual Model</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(individualResult).map(([name, d]) =>
                    <ResultBox key={name} title={name} data={d} />
                  )}
                </div>
              </>
            )}
          </div>
        )}
  
        {insights && (
          <div className="max-w-4xl mx-auto mt-12 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
            <h2 className="text-2xl font-semibold text-center text-indigo-300 mb-4">
              Model Diagnostics
            </h2>
            <ModelMetrics metrics={insights.metrics} />
            <FeatureBar title="Random Forest Importance" importances={insights.feature_importance.rfr} />
            <FeatureBar title="XGBoost Importance" importances={insights.feature_importance.xgb} />
          </div>
        )}
      </div>
    );
  }


function ResultBox({ title, data }) {
  const format = (val) => val !== undefined ? Number(val).toFixed(2) : '—';

  return (
    <div className="w-full sm:w-80 bg-[#1e2147] border border-indigo-400 rounded-lg p-4 text-center">
      <h3 className="text-xl font-bold text-indigo-200 mb-3">{title.toUpperCase()}</h3>
      <p><span className="font-medium">Predicted Points:</span> {format(data.predicted_points)}</p>
      <p><span className="font-medium">MAE:</span> {format(data.mae)}</p>
      <p><span className="font-medium">RMSE:</span> {format(data.rmse)}</p>
    </div>
  );
}

export default App;