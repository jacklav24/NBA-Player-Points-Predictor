export default function ResultBox({ title, data }) {
  const format = (val) =>
    typeof val === "number" && isFinite(val) ? val.toFixed(1) : "—";

  const labelMap = {
    rfr: "Random Forest",
    xgb: "XGBoost",
    lgb: "LightGBM",
    stk: "Stacked",
  };

  const displayTitle = labelMap[title.toLowerCase()] || title;

  return (
    <div className="sm:w-25 bg-[#1e2147] border border-indigo-400 rounded-lg px-4 py-3 text-center shadow">
      <div className="text-sm text-indigo-300 uppercase tracking-wide mb-2">
        {displayTitle}
      </div>
      <div className="text-4xl font-extrabold text-yellow-200">
        {format(data?.predicted_points)}
      </div>
    </div>
  );
}

