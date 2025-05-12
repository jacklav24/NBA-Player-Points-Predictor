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



import React, { useState } from "react";
import FeatureBar from "./FeatureBar"; 
import CombinedFeatureChart from "./CombinedFeatureChart";

export default function ModelMetrics({ metrics, featureImportance, playerPredicted, playerName }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const formatPlayerName = (name) =>
    name
      ?.split(/[_-]/)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(" ");


  const models = [
    { key: "rfr", label: "Random Forest" },
    { key: "xgb", label: "XGBoost" },
    { key: "lgb", label: "LGB" },
    { key: "stk", label: "Stacked"}
  ];
  const [selectedModels, setSelectedModels] = useState(models.map(m => m.key));

  const suffixes = ["_g", ...(playerPredicted ? ["_i", "_b"] : [])];
  const metricKeys = ["mae", "rmse", "r2", "bias", "within_n"];

  const toggleModel = (key) => {
    setSelectedModels(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  const visibleModels = models.filter(m => selectedModels.includes(m.key));

  
  const format = (val, metric) => {
    if (typeof val !== "number" || !isFinite(val)) return "—";
    if (metric === "within_n") return (val * 100).toFixed(0) + "%";
    return val.toFixed(2);
  };

  const getColorHSL = (value, values, metric) => {
    if (typeof value !== "number") return { bg: "hsl(220, 10%, 15%)", text: "#ccc" };

    const metricRanges = {
      r2: { min: 0, max: 1 },
      bias: null,
      mae: { min: 0, max: 6 },
      rmse: { min: 0, max: 8 },
      within_n: { min: 0, max: 1 },
    };

    const fixedRange = metricRanges[metric];
    let scale;

    if (metric === "bias") {
      const numericValues = values.map(v => (typeof v === 'number' ? Math.abs(v) : null)).filter(v => v !== null);
      const max = Math.max(...numericValues, 1);
      scale = Math.abs(value) / max;
    } else if (fixedRange) {
      const { min, max } = fixedRange;
      scale = (value - min) / (max - min);
      if (metric !== 'r2' && metric !== 'within_n') scale = 1 - scale;
    } else {
      const isHigherBetter = (metric === "r2" || metric === "within_n");
      const numericValues = values.filter(v => typeof v === 'number');
      const min = Math.min(...numericValues);
      const max = Math.max(...numericValues);
      const range = max - min || 1;
      scale = isHigherBetter ? (value - min) / range : (max - value) / range;
    }

    scale = Math.max(0, Math.min(1, scale));
    const hue = metric === 'bias' ? (1 - scale) * 120 : scale * 120;
    const saturation = 70;
    const lightness = 35 + scale * 20;
    const bg = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

    const [r, g, b] = hslToRgb(hue / 360, saturation / 100, lightness / 100);
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    const text = yiq >= 128 ? '#000' : '#fff';

    return { bg, text };
  };

  function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) r = g = b = l;
    else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }
    return [r * 255, g * 255, b * 255];
  }

  return (
    <div className="w-full mx-auto mt-12 p-6 bg-[#2a2d55] shadow-xl">
      <h2 className="text-2xl font-semibold text-center text-indigo-300 mb-4">
              Model Diagnostics {playerName && <span className="font-normal text-indigo-200"> — {formatPlayerName(playerName)}</span>}
            </h2>
      <hr className="border-indigo-400 mb-4" />

      <div className="flex justify-end mb-2">
        <div className="relative inline-block text-left">
          <button
            className="inline-flex justify-center w-full rounded-md border border-indigo-400 shadow-sm px-3 py-1 bg-[#1e2147] text-sm font-medium text-indigo-300 hover:bg-[#2a2d55]"
            type="button"
            onClick={() => setDropdownOpen(open => !open)}
          >
            Select Models
          </button>
          {dropdownOpen && (
            <div className="absolute right-0 z-10 mt-2 w-40 rounded-md shadow-lg bg-[#2a2d55] ring-1 ring-black ring-opacity-5">
              <div className="py-1">
                {models.map(({ key, label }) => (
                  <label key={key} className="flex items-center px-4 py-1 text-sm text-gray-200">
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(key)}
                      onChange={() => toggleModel(key)}
                      className="mr-2"
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="table-auto w-full text-sm text-gray-200 border-collapse">
          <thead>
            <tr>
              <th className="px-2 py-2"></th>
              {suffixes.map((suffix, idx) => (
                <th
                  key={suffix}
                  colSpan={visibleModels.length}
                  className={`px-4 py-2 text-center text-indigo-300 text-xl ${
                    idx < suffixes.length - 1 ? "border-r-2 border-indigo-400" : ""
                  }`}
                >
                  {"Global Individual Blended".split(" ")[idx]}
                </th>
                
              ))}
              
            </tr>
            <tr className="border-b-2 border-indigo-400">
              <th className="px-2 py-2"></th>
              {suffixes.flatMap((suffix, groupIdx) =>
                visibleModels.map(({ label }, modelIdx) => (
                  <th
                    key={`${suffix}-${label}`}
                    className={`px-2 py-1 text-center text-indigo-300 font-thin${
                      groupIdx < suffixes.length - 1 && modelIdx === visibleModels.length - 1
                        ? "border-r-2 border-indigo-400 font-thin"
                        : ""
                    }`}
                  >
                    {label}
                  </th>
                ))
              )}
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((metric) => {
              const rowValues = suffixes.flatMap(suffix =>
                visibleModels.map(({ key }) => metrics?.[`${key}_${metric}${suffix}`] ?? null)
              );

              return (
                <tr key={metric} className="border-t border-indigo-400">
                  <td className="px-2 py-3 text-gray-300 uppercase font-bold text-xl">{metric === "r2" ? "r²" : metric}</td>
                  {suffixes.flatMap(suffix =>
                    visibleModels.map(({ key }) => {
                      const raw = metrics?.[`${key}_${metric}${suffix}`] ?? null;
                      const hasData = raw !== null;
                      const val = hasData ? format(raw, metric) : "—";
                      const { bg, text } = getColorHSL(raw, rowValues, metric);
                      const isLastInGroup =
                      visibleModels.findIndex(m => m.key === key) === visibleModels.length - 1 &&
                      suffix !== suffixes[suffixes.length - 1];
                      return (
                        <td
                          key={`${key}_${suffix}_${metric}`}
                          className={`px-2 py-3 text-center ${isLastInGroup ? "border-r-2 border-indigo-400" : ""
                          }`}
                        >
                          <span
                            title={hasData && typeof raw === "number" ? raw.toFixed(4) : ""}
                            style={{
                              backgroundColor: bg,
                              color: text,
                              padding: "2px 6px",
                              borderRadius: "6px",
                              display: "inline-block",
                              minWidth: "40px",
                              transition: "background-color 0.3s",
                            }}
                          >
                            {val}
                          </span>
                        </td>
                      );
                    })
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <hr className="border-indigo-400 mt-4" />
      <CombinedFeatureChart featureImportance={featureImportance} />
    </div>
  );
}
