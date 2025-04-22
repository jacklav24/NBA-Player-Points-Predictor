import React from 'react';

export default function ModelMetrics({ metrics }) {
  return (
    <div className="bg-[#1e2147] p-4 rounded-lg mb-6 text-gray-200">
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>MAE: {metrics.mae}</div>
        <div>RMSE: {metrics.rmse}</div>
        <div>R²: {metrics.r2}</div>
        <div>Bias: {metrics.bias}</div>
      </div>
    </div>
  );
}
