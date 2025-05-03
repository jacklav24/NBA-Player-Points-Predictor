import React from 'react';
import CustomComboboxDropdown from './CustomComboboxDropdown';

/**
 * Props:
 * - teams, players, opponents, teamLabels
 * - filters: { team, player, opponent, location }
 * - onFilterChange: { setTeam, setPlayer, setOpponent, setLocation }
 * - controls: { handlePredict, kickOffOptimize, kickOffReTrain }
 * - status: { loading, reTuning, reTraining }
 * - saveRun: boolean
 * - onToggleSaveRun: () => void
 */
export default function RunControls({
  teams,
  players,
  opponents,
  teamLabels,
  handlePredict,
  filters: { team, player, opponent, location },
  onFilterChange: { setTeam, setPlayer, setOpponent, setLocation },
  controls: { kickOffOptimize, kickOffReTrain },
  status: { loading, reTuning, reTraining },
  saveRun,
  onToggleSaveRun
}) {
  return (
    <div className="w-full max-w-8xl mx-auto bg-[#2a2d55] p-6 rounded-xl shadow-lg">
      <div className="flex flex-wrap md:flex-nowrap justify-between items-end gap-4 mb-4">
        <CustomComboboxDropdown label="Team" options={teams} value={team} onChange={v => { setTeam(v); setPlayer(''); }} displayMap={teamLabels} minWidth={"180"} />
        <CustomComboboxDropdown label="Player" options={players} value={player} onChange={setPlayer} disabled={!team} minWidth={"180"} />
        <CustomComboboxDropdown label="Opponent" options={opponents} value={opponent} onChange={setOpponent} displayMap={teamLabels} minWidth={"180"}/> 
        <CustomComboboxDropdown label="Location" options={[ 'Home','Away' ]} value={location} onChange={setLocation} displayMap={{ Home: 'Home (H)', Away: 'Away (A)' }} minWidth={"100"} />

        <button
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px] min-w-[120px]"
          onClick={handlePredict}
          disabled={loading || reTuning}
        >
          {loading ? 'Predicting' : 'Predict'}
        </button>

        <button
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px] text-sm min-w-[185px]"
          onClick={kickOffOptimize}
          disabled={reTuning}
        >
          {reTuning ? 'Tuning' : 'Tune Hyperparameters'}
        </button>

        <button
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 h-[42px] text-sm min-w-[160px]"
          onClick={kickOffReTrain}
          disabled={reTraining}
        >
          {reTraining ? 'Training...' : 'Train Global Model'}
        </button>
        <button className={`px-3 py-2 rounded-lg h-[42px] min-w-[100px] transition ${saveRun ? 'bg-indigo-500 shadow-inner text-white' : 'bg-gray-700 text-white'} text-sm`} onClick={onToggleSaveRun}>
          Save Run
        </button>
      </div>
      {(loading || reTuning) && (
        <div className="text-center mt-2 animate-pulse">
          <span className="text-indigo-300">Calculating results...</span>
        </div>
      )}
    </div>
  

  );
}
