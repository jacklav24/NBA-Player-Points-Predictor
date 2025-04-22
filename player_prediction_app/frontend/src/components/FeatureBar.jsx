import React from 'react';

export default function FeatureBar({ title, importances }) {
  const sorted = Object.entries(importances)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);

  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold text-indigo-300 mb-2">{title}</h3>
      {sorted.map(([feat, imp]) => (
        <div key={feat} className="flex items-center mb-1">
          <span className="w-32 text-xs">{feat}</span>
          <div
            className="h-2 bg-indigo-500 mx-2"
            style={{ width: `${Math.min(imp * 200, 200)}px` }}
          />
          <span className="text-xs">{imp.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
