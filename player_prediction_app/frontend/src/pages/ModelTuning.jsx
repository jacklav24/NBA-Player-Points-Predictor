import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function ModelTuning() {
    const [bestParams, setBestParams] = useState([]);
    const [trials, setTrials] = useState([]);

    
    useEffect(() => {
        fetch("http://localhost:8000/tuning_results")
          .then(res => res.json())
          .then(data => {
            setBestParams(data.best_params);
            setTrials(data.trials);
          });
      }, []);
}