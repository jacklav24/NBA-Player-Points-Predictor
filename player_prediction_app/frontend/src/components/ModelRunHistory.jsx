
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ModelRunHistory({ runs = [], initialCount = 5, onSelectRun }) {
  const navigate = useNavigate();
  const [showAll, setShowAll] = useState(false);
  const [openSections, setOpenSections] = useState({});
  const [parsedRuns, setParsedRuns] = useState([]);

  // Parse and sort runs
  useEffect(() => {
    const safeParse = str => {
      try { return JSON.parse(str.replace(/'/g, '"')); }
      catch { return []; }
    };
    const pr = (runs || []).map(r => ({
      id: r.run_id || r.id,
      date: r.timestamp || r.date,
      features: typeof r.features === 'string' ? safeParse(r.features) : r.features || [],
      scaled_features: typeof r.scaled_features === 'string' ? safeParse(r.scaled_features) : r.scaled_features || [],
      metrics: typeof r.metrics === 'string' ? safeParse(r.metrics) : r.metrics || {}
    }));
    pr.sort((a, b) => new Date(b.date) - new Date(a.date));
    setParsedRuns(pr);
  }, [runs]);

  const displayed = showAll ? parsedRuns : parsedRuns.slice(0, initialCount);
  const MODEL_KEYS = ['rfr', 'xgb', 'stacked'];
  const METRICS = ['mae', 'rmse', 'r2', 'bias'];

  const toggleSection = (runId, section) => {
    setOpenSections(prev => ({
      ...prev,
      [runId]: { ...prev[runId], [section]: !prev[runId]?.[section] }
    }));
  };

  const metricColor = metric => ['r2','bias'].includes(metric)
    ? 'text-indigo-400' : 'text-green-300';
  const displayMetric = metric => metric === 'r2' ? 'r²' : metric.toUpperCase();

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
        const date = new Date(run.date);
        const dateLabel = isNaN(date.getTime())
          ? run.date
          : `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;

        return (
          <div key={run.id} className="bg-[#2a2d55] border border-indigo-400 p-4 rounded-lg relative">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-300">Run: {dateLabel}</span>
              <div className="space-x-4 text-sm">
                <button onClick={() => toggleSection(run.id, 'metrics')} className="text-indigo-200 hover:underline">
                  {open.metrics ? 'Hide Metrics' : 'Show Metrics'}
                </button>
                <button onClick={() => toggleSection(run.id, 'features')} className="text-indigo-200 hover:underline">
                  {open.features ? 'Hide Features' : 'Features'}
                </button>
                <button onClick={() => toggleSection(run.id, 'scaled')} className="text-indigo-200 hover:underline">
                  {open.scaled ? 'Hide Scaled' : 'Scaled'}
                </button>
              </div>
            </div>

            {/* metrics grid */}
            <div className="grid grid-cols-3 gap-6 text-sm mb-2">
              {MODEL_KEYS.map(key => (
                <div key={key} className="space-y-1">
                  <span className="font-medium text-indigo-200 uppercase">{key}</span>
                  {(open.metrics ? METRICS : ['mae','rmse']).map(metric => {
                    const val = run.metrics[`${key}_${metric}_g`];
                    const displayVal = val != null ? val.toFixed(2) : '—';
                    return (
                      <div key={metric} className="flex items-center">
                        <span className="text-white">{displayMetric(metric)}:</span>
                        <span className={`${metricColor(metric)} ml-1`}>{displayVal}</span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* features */}
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
              <div className="mb-2">
                <h4 className="text-sm font-medium text-indigo-200 mb-1">Scaled Features:</h4>
                <ul className="grid grid-cols-2 gap-x-4 text-sm text-gray-200 list-disc list-inside">
                  {run.scaled_features.map((f, idx) => <li key={`${run.id}-scaled-${idx}`}>{f}</li>)}
                </ul>
              </div>
            )}

            {/* full diagnostics button */}
            <div className="flex justify-end mt-4">
              <button
                onClick={() => (onSelectRun ? onSelectRun(run) : navigate(`/runs/${run.id}`))}
                className="bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded-lg text-sm"
              >
                View Full Run Diagnostics
              </button>
            </div>
          </div>
        );
      })}

      {displayed.length === 0 && (
        <p className="text-center text-gray-400">No past runs to show.</p>
      )}
    </div>
  );
}
