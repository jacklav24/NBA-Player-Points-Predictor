import React from "react";

import FeatureBar from "./FeatureBar"; // make sure the path is correct

export default function ModelMetrics({ metrics, featureImportance, playerPredicted }) {
  const format = (val) =>
    typeof val === "number" && !isNaN(val) ? val.toFixed(2) : "—";

 
  const renderBlock = (title, prefix = "", suffix = "", isIndividual = false) => {
    const hasData = !isIndividual || playerPredicted;
  
    return (
      <div className="flex-1 bg-[#1e2147] p-4 rounded-lg shadow-md border border-indigo-400 min-h-[120px] flex flex-col justify-between">
        <h4 className="text-lg font-semibold text-indigo-200 mb-2">{title}</h4>
        {hasData ? (
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-200">
            <div>
              <span className="font-medium">MAE:</span>{" "}
              <span className="text-yellow-300">{format(metrics[`${prefix}mae${suffix}`])}</span>
            </div>
            <div>
              <span className="font-medium">RMSE:</span>{" "}
              <span className="text-yellow-300">{format(metrics[`${prefix}rmse${suffix}`])}</span>
            </div>
            <div>
              <span className="font-medium">R²:</span>{" "}
              <span className="text-yellow-300">{format(metrics[`${prefix}r2${suffix}`])}</span>
            </div>
            <div>
              <span className="font-medium">Bias:</span>{" "}
              <span className="text-yellow-300">{format(metrics[`${prefix}bias${suffix}`])}</span>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-400 italic">Select a player to see individual results.</p>
        )}
      </div>
    );
  };
    
  


  return (
    <div className="space-y-8">
      {/* Global + Individual Metrics */}
      <div className="flex flex-col md:flex-row md:space-x-6">
        {renderBlock("Random Forest (Global)", "rfr_", "_g")}
        {renderBlock("Random Forest (Individual)", "rfr_", "_i", true)}
      </div>
      <div className="flex flex-col md:flex-row md:space-x-6">
        {renderBlock("XGBoost (Global)", "xgb_", "_g")}
        {renderBlock("XGBoost (Individual)", "xgb_","_i", true)}
      </div>
      <div className="flex flex-col md:flex-row md:space-x-6">
        {renderBlock("Stacked (Global)", "stacked_", "_g")}
        {renderBlock("Stacked (Individual)", "stacked_", "_i", true)}
      </div>

      {/* Feature Importance Section */}
      <div className="flex flex-col md:flex-row md:space-x-6">
        <div className="flex-1 border border-indigo-400 rounded-lg p-4 bg-[#1e2147]">
          <FeatureBar title="Random Forest Importance" importances={featureImportance?.rfr || {}} />
        </div>
        <div className="flex-1 border border-indigo-400 rounded-lg p-4 bg-[#1e2147] mt-6 md:mt-0">
          <FeatureBar title="XGBoost Importance" importances={featureImportance?.xgb || {}} />
        </div>
      </div>
    </div>
  );
}



