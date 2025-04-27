import React, { useMemo } from 'react';

const MODELS = [
  { key: 'rfr_g', label: 'RF (Global)' },
  { key: 'xgb_g', label: 'XGB (Global)' },
  { key: 'rfr_i', label: 'RF (Individual)' },
  { key: 'xgb_i', label: 'XGBoost (Individual)' },
];

export default function CombinedFeatureChart({ featureImportance = {} }) {
  // Collect & sort features by descending max importance
  const { features, maxImp } = useMemo(() => {
    const featSet = new Set();
    MODELS.forEach(m =>
      Object.keys(featureImportance[m.key] || {}).forEach(f => featSet.add(f))
    );
    const feats = Array.from(featSet);
    const perMax = feats.map(f =>
      Math.max(...MODELS.map(m => featureImportance[m.key]?.[f] || 0))
    );
    feats.sort((a, b) => perMax[feats.indexOf(b)] - perMax[feats.indexOf(a)]);
    return { features: feats, maxImp: Math.max(...perMax) };
  }, [featureImportance]);

  return (
    <div
      className="bg-[#1e2147] border border-indigo-400 rounded-lg overflow-auto"
      style={{ maxHeight: '70vh' }}
    >
      <table className="min-w-full table-fixed border-collapse">
        <thead className="sticky top-0 bg-[#1e2147]">
          <tr>
            <th className="px-4 py-2 text-left text-indigo-300">Feature Importance</th>
            {MODELS.map((m, idx) => (
              <th
                key={m.key}
                className={`px-4 py-2 text-left text-indigo-300 ${
                  idx < MODELS.length - 1 ? 'border-r border-indigo-600' : ''
                }`}
              >
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-indigo-600">
          {features.map(feat => (
            <tr key={feat} className="group hover:bg-[#2a2e52] transition-colors">
              <td className="px-4 py-2 text-white whitespace-nowrap border-r border-indigo-600">
                {feat}
              </td>
              {MODELS.map((m, idx) => {
                const imp = featureImportance[m.key]?.[feat] || 0;
                const widthPct = maxImp ? (imp / maxImp) * 100 : 0;
                return (
                  <td
                    key={m.key}
                    className={`px-2 py-2 align-middle ${
                      idx < MODELS.length - 1 ? 'border-r border-indigo-600' : ''
                    }`}
                  >
                    <div className="relative">
                      <div
                        className="h-4 bg-indigo-500 rounded transition-colors duration-150 group-hover:bg-indigo-400"
                        style={{ width: `${widthPct}%` }}
                      />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[#1e2147] text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity pointer-events-none">
                        {imp.toFixed(3)}
                      </div>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
