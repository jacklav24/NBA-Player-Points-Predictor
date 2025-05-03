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


import React from 'react';
import ResultBox from './ResultBox';

export default function Predictions({ global, indiv, blended, playerName }) {
  const formatPlayerName = (name) =>
    name
      ?.split(/[_-]/)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(" ");

  const hasPredictions = global || indiv || blended;

  const sections = [
    { label: "Global", data: global },
    { label: "Individual", data: indiv },
    { label: "Blended", data: blended },
  ];

  return (
    <div className="w-full max-w-7xl mx-auto mt-12 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
      <h2 className="text-3xl font-semibold text-indigo-300 mb-4">
  Predictions
  {playerName && hasPredictions && (
    <span className="font-normal text-indigo-200"> — {formatPlayerName(playerName)}</span>
  )}
</h2>
      
      

      {!hasPredictions && (
        <div className="text-center mt-2">
          <span className="text-indigo-300 text-sm">
            (Make a Player and Opponent Selection to Get Prediction)
          </span>
        </div>
      )}

      {sections.map(({ label, data }) =>
        data ? (
          <div key={label} className="mb-6">
            <h3 className="text-center text-indigo-300 uppercase mb-2 ml-1">
              {label}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(data)
                .filter(([key]) => key !== "alpha")
                .map(([name, val]) => (
                  <ResultBox
                    key={`${label}-${name}`}
                    title={name}
                    data={
                      typeof val === "number"
                        ? { predicted_points: val, mae: "-", rmse: "-" }
                        : val
                    }
                    playerName={playerName}
                  />
                ))}
            </div>
          </div>
        ) : null
      )}
    </div>
  );
}
