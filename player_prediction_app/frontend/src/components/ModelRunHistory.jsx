import React, { useState, useEffect } from 'react';

export default function ModelRunHistory({ runs = [], initialCount = 5 }) {
  const [showAll, setShowAll] = useState(false);
  const [openSections, setOpenSections] = useState({});
  const [parsedRuns, setParsedRuns] = useState([]);

  // Parse runs strings into proper objects and sort by timestamp descending
  useEffect(() => {
    const safeParse = str => {
      try {
        return JSON.parse(str.replace(/'/g, '"'));
      } catch {
        return [];
      }
    };

    const pr = (runs || []).map(r => ({
      id: r.run_id,
      date: r.timestamp,
      features: typeof r.features === 'string' ? safeParse(r.features) : r.features,
      scaled_features: typeof r.scaled_features === 'string' ? safeParse(r.scaled_features) : r.scaled_features,
      metrics: typeof r.metrics === 'string' ? safeParse(r.metrics) : r.metrics || {}
    }));

    // sort descending by date (newest first)
    pr.sort((a, b) => new Date(b.date) - new Date(a.date));
    setParsedRuns(pr);
  }, [runs]);

  const displayed = showAll ? parsedRuns : parsedRuns.slice(0, initialCount);
  const MODEL_KEYS = ['rfr', 'xgb', 'stacked'];
  const METRICS = ['mae', 'rmse', 'r2', 'bias'];

  const toggleSection = (runId, section) => {
    setOpenSections(prev => ({
      ...prev,
      [runId]: {
        ...prev[runId],
        [section]: !prev[runId]?.[section]
      }
    }));
  };

  // Color classes for each metric type
  const metricColor = (metric) => {
    switch(metric) {
      case 'mae': return 'text-green-300';
      case 'rmse': return 'text-green-300';
      case 'r2': return 'text-indigo-400';
      case 'bias': return 'text-indigo-400';
      default: return 'text-gray-200';
    }
  };

  // Helper for display name, show r2 as r²
  const displayMetric = (metric) => metric === 'r2' ? 'r²' : metric.toUpperCase();

  return (
    <div className="max-w-4xl mx-auto mt-8 space-y-6">
      <h2 className="text-2xl font-semibold text-center text-indigo-300">Recent Model Runs</h2>
      <div className="flex justify-end">
        {parsedRuns.length > initialCount && (
          <button
            className="text-sm text-indigo-200 hover:underline"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? 'Show Less' : `Show All (${parsedRuns.length})`}
          </button>
        )}
      </div>

      {displayed.map(run => {
        const open = openSections[run.id] || {};
        return (
          <div key={run.id} className="bg-[#2a2d55] border border-indigo-400 p-4 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-300">
                Run: {run.date.split(' ')[0]} {new Date(run.date).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', hour12: true })}
              </span>
              <div className="space-x-4 text-sm">
                <button
                  className="text-indigo-200 hover:underline"
                  onClick={() => toggleSection(run.id, 'metrics')}
                >
                  {open.metrics ? 'Hide Metrics' : 'Show Metrics'}
                </button>
                <button
                  className="text-indigo-200 hover:underline"
                  onClick={() => toggleSection(run.id, 'features')}
                >
                  {open.features ? 'Hide Features' : 'Features'}
                </button>
                <button
                  className="text-indigo-200 hover:underline"
                  onClick={() => toggleSection(run.id, 'scaled')}
                >
                  {open.scaled ? 'Hide Scaled' : 'Scaled'}
                </button>
              </div>
            </div>

            {/* summary metrics for all models */}
            {!open.metrics && (
              <div className="grid grid-cols-3 gap-6 text-sm text-gray-200 mb-2">
                {MODEL_KEYS.map(key => (
                  <div key={key} className="space-y-1">
                    <span className="font-medium text-indigo-200 uppercase">{key}</span>
                    <div>
                      MAE: <span className={metricColor('mae')}>{run.metrics[`${key}_mae_g`]?.toFixed(2) ?? '—'}</span>
                    </div>
                    <div>
                      RMSE: <span className={metricColor('rmse')}>{run.metrics[`${key}_rmse_g`]?.toFixed(2) ?? '—'}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* full metrics for all models */}
            {open.metrics && (
              <div className="grid grid-cols-3 gap-6 text-sm text-gray-200 mb-2">
                {MODEL_KEYS.map(key => (
                  <div key={key} className="space-y-1">
                    <span className="font-medium text-indigo-200 uppercase">{key}</span>
                    {METRICS.map(metric => (
                      <div key={metric}>
                        {displayMetric(metric)}:{' '}
                        <span className={metricColor(metric)}>
                          {run.metrics[`${key}_${metric}_g`]?.toFixed(2) ?? '—'}
                        </span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}

            {/* feature list */}
            {open.features && (
              <div className="mb-2">
                <h4 className="text-sm font-medium text-indigo-200 mb-1">Features:</h4>
                <ul className="grid grid-cols-2 gap-x-4 text-sm text-gray-200 list-disc list-inside">
                  {run.features.map((f, idx) => <li key={`${run.id}-feature-${idx}`}>{f}</li>)}
                </ul>
              </div>
            )}

            {/* scaled features */}
            {open.scaled && (
              <div>
                <h4 className="text-sm font-medium text-indigo-200 mb-1">Scaled Features:</h4>
                <ul className="grid grid-cols-2 gap-x-4 text-sm text-gray-200 list-disc list-inside">
                  {run.scaled_features.map((f, idx) => <li key={`${run.id}-scaled-${idx}`}>{f}</li>)}
                </ul>
              </div>
            )}
          </div>
        );
      })}

      {displayed.length === 0 && (
        <p className="text-center text-gray-400">No past runs to show.</p>
      )}
    </div>
  );
}


