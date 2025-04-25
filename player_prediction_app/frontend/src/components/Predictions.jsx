import React, { useEffect, useState } from 'react';
import ResultBox from './ResultBox';




export default function Predictions({indiv, global, playerName}) {
    const [globalResult, setGlobalResult] = useState(global);
    const [individualResult, setIndividualResult] = useState(indiv);

    useEffect(() => {
      setGlobalResult(global);
    }, [global]);
  
    useEffect(() => {
      setIndividualResult(indiv);
    }, [indiv]);
  return (
    <div>
    {(globalResult || individualResult) && (
      <div className="max-w-7xl mx-auto mt-8 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
        <h2 className="text-2xl font-semibold mb-4 text-center text-indigo-200">
          Prediction Results
        </h2>

        {globalResult && (
          <>
            <h3 className="text-xl font-medium mb-4 text-center">Global Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(globalResult).map(([name, d]) =>
                <ResultBox key={name} title={name} data={d} playerName={playerName}/>
              )}
            </div>
          </>
        )}

        {individualResult && (
          <>
            <h3 className="text-xl font-medium mb-4 text-center">Individual Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(individualResult).map(([name, d]) =>
                <ResultBox key={name} title={name} data={d} playerName={playerName} />
              )}
            </div>
          </>
        )}
      </div>
    )}
    </div>);
}