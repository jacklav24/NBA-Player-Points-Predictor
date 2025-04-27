import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import ModelRunHistory from '../components/ModelRunHistory';

export default function RunHistoryPage() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/get_runs').then(res => setRuns(res.data)).catch(console.error);
  }, []);

  return (
    <div className="p-6">
      <button
        onClick={() => navigate('/')}
        className="absolute top-6 right-6 bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded-lg text-sm"
      >
        Back to Home
      </button>
      <h2 className="text-2xl font-bold mb-4">Run History</h2>
      <ModelRunHistory runs={runs} onSelectRun={run => navigate(`/runs/${run.id}`)} />
    </div>
  );
}

