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


import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams, Link, useNavigate } from 'react-router-dom';
import ModelMetrics from '../components/ModelMetrics';
import Predictions from '../components/Predictions';

import CombinedFeatureChart from '../components/CombinedFeatureChart';

export default function RunDetailsPage() {
  const { runId } = useParams();
  const [insights, setInsights] = useState(null);
  const navigate = useNavigate();
  

  useEffect(() => {
    axios.get(`http://localhost:8000/run-history/${runId}`)
      .then(res => setInsights(res.data), console.log(insights))
      .catch(console.error);
  }, [runId]);

  if (!insights) return <div className="p-6">Loading...</div>;

  return (
    <div className="p-6">
      <button
        onClick={() => navigate('/')}
        className="absolute top-6 right-6 bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded-lg text-sm"
      >
        Back to Home
      </button>
      <Link to="/runs" className="text-indigo-400 hover:underline mb-4 block">← Back to History</Link>
      <h2 className="text-2xl font-bold mb-4">Run Details: {insights.name}</h2>
      <div className="space-y-6">
        <ModelMetrics metrics={insights.metrics} playerPredicted={insights.playerPredicted} featureImportance={insights.feature_importance} />
        
      </div>
    </div>
  );
}