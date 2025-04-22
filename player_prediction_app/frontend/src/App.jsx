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
import { Combobox, ComboboxInput, ComboboxOption, ComboboxOptions, ComboboxButton } from '@headlessui/react';
import { CheckIcon, ChevronDownIcon } from '@heroicons/react/24/solid'
import teamLabels from './constants/teamLabels';
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
        }),
        axios.post('http://localhost:8000/predict', {
          player_name: player,
          team: team,
          opponent: opponent,
        })
      ]);

      setGlobalResult(globalRes.data);
      setIndividualResult(individualRes.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Check console for details.");
    } finally {
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
            <ComboboxDropdown label="Team" options={teams} value={team} onChange={v => { setTeam(v); setPlayer(''); }} />
            <ComboboxDropdown label="Player" options={players} value={player} onChange={setPlayer} disabled={!team} />
            <ComboboxDropdown label="Opponent" options={opponents} value={opponent} onChange={setOpponent} />
  
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


  // return (
  //   <div className="min-h-screen bg-[#1e2147] text-[#f5f5f5] p-8">
  //     <h1 className="text-3xl font-bold mb-8 text-center">NBA Player Points Predictor</h1>

  //     <div className="max-w-6xl mx-auto bg-[#2a2d55] p-6 rounded-xl shadow-lg">
  //       <div className="flex flex-wrap md:flex-nowrap justify-between items-end gap-4 mb-4">
  //         <ComboboxDropdown label="Team" options={teams} value={team} onChange={(val) => { setTeam(val); setPlayer(''); }} />
  //         <ComboboxDropdown label="Player" options={players} value={player} onChange={setPlayer} disabled={!team} />
  //         <ComboboxDropdown label="Opponent" options={opponents} value={opponent} onChange={setOpponent} />

  //         <button
  //           className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg transition disabled:opacity-50 h-[42px]"
  //           onClick={handlePredict}
  //           disabled={loading}
  //         >
  //           {loading ? 'Predicting...' : 'Predict'}
  //         </button>
  //       </div>

  //       {loading && (
  //         <div className="text-center mt-2 animate-pulse">
  //           <span className="text-indigo-300">Calculating results...</span>
  //         </div>
  //       )}
  //     </div>

  //     {(globalResult || individualResult) && (
  //       <div className="max-w-7xl mx-auto mt-8 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
  //         <h2 className="text-2xl font-semibold mb-4 text-center">Prediction Results</h2>

  //         {globalResult && (
  //           <>
  //             <h3 className="text-xl font-medium mb-4 text-center text-indigo-200">Global Model</h3>
  //             <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 justify-items-center">
  //               {Object.entries(globalResult).map(([modelName, data]) => (
  //                 <ResultBox key={modelName} title={modelName} data={data} />
  //               ))}
  //             </div>
  //           </>
  //         )}

  //         {individualResult && (
  //           <>
  //             <h3 className="text-xl font-medium mb-4 text-center text-indigo-200">Individual Player Model</h3>
  //             <div className="grid grid-cols-1 md:grid-cols-3 gap-4 justify-items-center">
  //               {Object.entries(individualResult).map(([modelName, data]) => (
  //                 <ResultBox key={modelName} title={modelName} data={data} />
  //               ))}
  //             </div>
  //           </>
  //         )}
  //       </div>
  //     )}
  //   </div>
  // );


function ComboboxDropdown({ label, options, value, onChange, disabled = false }) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);

  const filtered = query === ''
  ? options
  : options.filter(opt => {
      const label = teamLabels[opt] || opt;
      return label.toLowerCase().includes(query.toLowerCase());
    });

  return (
    <div className="flex-1 min-w-[150px] max-w-[300px]">
      <Combobox value={value} onChange={onChange} disabled={disabled}>
        {/* <ComboboxLabel className="block text-sm font-medium text-gray-300 mb-1">{label}</ComboboxLabel> */}
        <div className="relative">
          <ComboboxInput
            className="w-full px-3 py-2 rounded border border-gray-600 bg-[#1e2147] text-[#f5f5f5] focus:outline-none focus:ring-2 focus:ring-indigo-400"
            displayValue={(val) => teamLabels[val] || val}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setOpen(true)}
            onBlur={() => setTimeout(() => setOpen(false), 100)} // avoid closing before click registers
            placeholder={`Select ${label}`}
          />
          <ComboboxButton className="group absolute inset-y-0 right-0 px-2.5">
            <ChevronDownIcon className="size-4 fill-white/60 group-data-[hover]:fill-white" />
          </ComboboxButton>
          {open && filtered.length > 0 && (
            <ComboboxOptions className="absolute mt-1 w-full max-h-60 overflow-auto rounded-md bg-[#2a2d55] py-1 text-sm shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
              {filtered.map((opt) => (
                <ComboboxOption
                key={opt}
                value={opt}
                className="cursor-pointer select-none px-4 py-2 ui-active:bg-indigo-600 ui-active:text-white text-[#f5f5f5]"
              >
                {teamLabels[opt] || opt}
                </ComboboxOption>
              ))}
            </ComboboxOptions>
          )}
        </div>
      </Combobox>
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