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

function App() {
  const [allPlayers, setAllPlayers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [opponents, setOpponents] = useState([]);

  const [player, setPlayer] = useState('');
  const [team, setTeam] = useState('');
  const [opponent, setOpponent] = useState('');

  const [globalResult, setGlobalResult] = useState(null);
  const [individualResult, setIndividualResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    axios.get('http://localhost:8000/teams').then(res => setTeams(res.data));
    axios.get('http://localhost:8000/opponents').then(res => setOpponents(res.data));
  }, []);

  useEffect(() => {
    if (team) {
      axios.get(`http://localhost:8000/${team}/players`)
        .then(res => setAllPlayers(res.data))
        .catch(err => {
          console.error("Error fetching players for team", team, err);
          setAllPlayers([]);
        });
    } else {
      setAllPlayers([]);
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

      const [globalRes, indivRes] = await Promise.all([
        axios.post('http://localhost:8000/global_predict', {
          player_name: player,
          team,
          opponent
        }),
        axios.post('http://localhost:8000/predict', {
          player_name: player,
          team,
          opponent
        })
      ]);

      setGlobalResult(globalRes.data);
      setIndividualResult(indivRes.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#1e2147] text-[#f5f5f5] p-8">
      <h1 className="text-3xl font-bold mb-8 text-center">NBA Player Points Predictor</h1>

      <div className="max-w-6xl mx-auto bg-[#2a2d55] p-6 rounded-xl shadow-lg">
        <div className="flex flex-wrap md:flex-nowrap justify-between items-end gap-4 mb-4">
          <Dropdown label="Team" options={teams} value={team} onChange={(value) => { setTeam(value); setPlayer(''); }} />
          <Dropdown label="Player" options={allPlayers} value={player} onChange={setPlayer} disabled={!team} />
          <Dropdown label="Opponent" options={opponents} value={opponent} onChange={setOpponent} />
          <button
            className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg transition disabled:opacity-50 h-[42px]"
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
          <h2 className="text-2xl font-semibold mb-4 text-center">Prediction Results</h2>

          {globalResult && (
            <>
              <h3 className="text-xl font-medium mb-4 text-center text-indigo-200">Global Model</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 justify-items-center">
                {Object.entries(globalResult).map(([modelName, data]) => (
                  <ResultBox key={modelName} title={modelName} data={data} />
                ))}
              </div>
            </>
          )}

          {individualResult && (
            <>
              <h3 className="text-xl font-medium mb-4 text-center text-indigo-200">Individual Player Model</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 justify-items-center">
                {Object.entries(individualResult).map(([modelName, data]) => (
                  <ResultBox key={modelName} title={modelName} data={data} />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Dropdown({ label, options, value, onChange, disabled = false }) {
  return (
    <div className="flex-1 min-w-[150px] max-w-[200px]">
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <select
        disabled={disabled}
        className="w-full border border-gray-600 bg-[#1e2147] text-[#f5f5f5] rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select {label}</option>
        {options.map(opt => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
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
