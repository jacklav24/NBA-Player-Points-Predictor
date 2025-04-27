import React from "react";
import FeatureBar from "./FeatureBar"; 
import CombinedFeatureChart from "./CombinedFeatureChart";

export default function ModelMetrics({ metrics, featureImportance, playerPredicted, }) {
  // const format = (val) =>
  //   typeof val === "number" && isFinite(val) ? val.toFixed(2) : "—";
  const format = (val, metric) => {
    if (typeof val !== "number" || !isFinite(val)) return "—";
    if (metric === "within_n") return (val * 100).toFixed(0) + "%"; // 0.87 -> "87%"
    return val.toFixed(2);
  };
  const models = [
    { key: "rfr", label: "Random Forest" },
    { key: "xgb", label: "XGBoost" },
    { key: "stacked", label: "Stacked" },
  ];
  // const metricKeys = ["mae", "rmse", "r2", "bias"];
  const metricKeys = ["mae", "rmse", "r2", "bias", "within_n"];
  const suffixes = ["_g", "_i", "_b"];




  const getColorHSL = (value, values, metric) => {
    if (typeof value !== 'number') {
      return { bg: 'hsl(220, 10%, 15%)', text: '#ccc' };
    }
  
    // Define fixed reference ranges for specific metrics
    const metricRanges = {
      r2: { min: 0, max: 1 },
      bias: null, // dynamic (around 0)
      mae: { min: 0, max: 6 }, // Assume 0-20 reasonable error range (adjust)
      rmse: { min: 0, max: 8 }, // Same
      within_n: { min: 0, max: 1 }, // 0% to 100%
    };
  
    const fixedRange = metricRanges[metric];
  
    let scale;
    if (metric === "bias") {
      const numericValues = values.map(v => (typeof v === 'number' ? Math.abs(v) : null)).filter(v => v !== null);
      const max = Math.max(...numericValues, 1); // prevent 0-div
      scale = Math.abs(value) / max;
    } else if (fixedRange) {
      const { min, max } = fixedRange;
      scale = (value - min) / (max - min);
      if (metric !== 'r2' && metric !== 'within_n') {
        scale = 1 - scale; // For mae/rmse lower=better
      }
    } else {
      // fallback dynamic scaling
      const isHigherBetter = (metric === "r2" || metric === "within_n");
      const numericValues = values.filter(v => typeof v === 'number');
      if (numericValues.length === 0) {
        return { bg: 'hsl(220, 10%, 15%)', text: '#ccc' };
      }
      const min = Math.min(...numericValues);
      const max = Math.max(...numericValues);
      const range = max - min || 1;
      scale = isHigherBetter
        ? (value - min) / range
        : (max - value) / range;
    }
  
    // clamp scale
    scale = Math.max(0, Math.min(1, scale));
  
    // Color mapping
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
    if (s === 0) {
      r = g = b = l;
    } else {
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
    <div className="max-w-8xl mx-auto mt-8">
      <hr className="border-indigo-400 mb-4" />
      <div className="overflow-x-auto">
        <table className="table-auto w-full text-sm text-gray-200 border-collapse">
          <thead>
            <tr>
              <th className="px-2 py-2"></th>
              {models.map(({ label }, idx) => (
                <th
                  key={label}
                  colSpan={3}
                  className={`px-4 py-2 text-center text-indigo-300 ${
                    idx < models.length - 1 ? "border-r-2 border-indigo-400" : ""
                  }`}
                >
                  {label}
                </th>
              ))}
            </tr>
            <tr className="border-b-2 border-indigo-400">
              <th className="px-2 py-2"></th>
              {models.flatMap(({ key }, modelIdx) =>
                suffixes.map((_, i) => (
                  <th
                    key={key + i}
                    className={`px-2 py-1 text-center text-indigo-300 ${
                      modelIdx < models.length - 1 && i === suffixes.length - 1
                        ? "border-r-2 border-indigo-400"
                        : ""
                    }`}
                  >
                    {["Global", "Individual", "Blended"][i]}
                  </th>
                ))
              )}
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((metric) => {
              const rowValues = models.flatMap(({ key }) =>
                suffixes.map((suffix) => {
                  const hasData = suffix === "_g" || playerPredicted;
                  const raw = metrics[`${key}_${metric}${suffix}`];
                  return hasData && typeof raw === "number" ? raw : null;
                })
              );

              return (
                <tr key={metric} className="border-t border-indigo-400">
                  <td className="px-2 py-3 text-gray-300 uppercase font-bold text-xl">{metric === "r2" ? "r²" : metric}</td>
                  {models.flatMap(({ key }, modelIdx) =>
                    suffixes.map((suffix, suffixIdx) => {
                      const hasData = suffix === "_g" || playerPredicted;
                      // const raw = metrics[`${key}_${metric}${suffix}`];
                      const raw = metrics?.[`${key}_${metric}${suffix}`] ?? null;
                      const val = hasData ? format(raw, metric) : "—";
                      const { bg, text } = getColorHSL(
                        hasData ? raw : null,
                        rowValues,
                        metric
                      );
                      return (
                        <td
                          key={key + metric + suffix}
                          className={`px-2 py-3 text-center ${
                            modelIdx < models.length - 1 && suffixIdx === suffixes.length - 1
                              ? "border-r-2 border-indigo-400"
                              : ""
                          }`}
                        >
                          <span
                            title={
                              hasData && typeof raw === "number" ? raw.toFixed(4) : ""
                            }
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
      {/* Feature Importance Section */}
      {/* <div className="flex flex-col py-8 md:flex-row md:space-x-6">
        <div className="flex-1 border border-indigo-400 rounded-lg p-4 bg-[#1e2147]">
          <FeatureBar title="Global Random Forest Importance" importances={featureImportance?.rfr_g || {}} />
        </div>
        <div className="flex-1 border border-indigo-400 rounded-lg p-4 bg-[#1e2147] mt-6 md:mt-0">
          <FeatureBar title="Global XGBoost Importance" importances={featureImportance?.xgb_g || {}} />
        </div>
        <div className="flex-1 border border-indigo-400 rounded-lg p-4 bg-[#1e2147]">
          <FeatureBar title="Individual Random Forest Importance" importances={featureImportance?.rfr_i || {}} />
        </div>
        <div className="flex-1 border border-indigo-400 rounded-lg p-4 bg-[#1e2147] mt-6 md:mt-0">
          <FeatureBar title="Individual XGBoost Importance" importances={featureImportance?.xgb_i || {}} />
        </div>
        
      </div> */}
      <CombinedFeatureChart featureImportance={featureImportance} />
    </div>
  );
}
