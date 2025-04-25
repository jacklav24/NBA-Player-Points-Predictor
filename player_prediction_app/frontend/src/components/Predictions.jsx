import React from 'react';
import ResultBox from './ResultBox';

export default function Predictions({ global, indiv, blended, playerName }) {
  return (
    <>
     
      <div className="max-w-7xl mx-auto mt-8 p-6 bg-[#2a2d55] rounded-xl shadow-xl">
      <h2 className="text-3xl font-semibold text-center text-indigo-300 mb-4">
                Predictions
              </h2>
        {(!global && !indiv && !blended) &&
          <div className="text-center mt-2">
            <span className="text-indigo-300 text-m">(Make a Player and Opponent Selection to Get Prediction)</span>
          </div>
        }
        {global && (
          <>
            <h3 className="text-2xl font-semibold text-center text-indigo-300 mb-4">Global Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {Object.entries(global).map(([name, data]) => (
                <ResultBox
                  key={name}
                  title={name}
                  data={data}
                  playerName={playerName}
                />
              ))}
            </div>
          </>
        )}

        {indiv && (
          <>
            <h3 className="text-2xl font-semibold text-center text-indigo-300 mb-4">Individual Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {Object.entries(indiv).map(([name, data]) => (
                <ResultBox
                  key={name}
                  title={name}
                  data={data}
                  playerName={playerName}
                />
              ))}
            </div>
          </>
        )}

        {blended && (
          <>
            <h3 className="text-2xl font-semibold text-center text-indigo-300 mb-4">Blended Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(blended)
                .filter(([key]) => key !== 'alpha')
                .map(([name, predicted_points]) => (
                  <ResultBox
                    key={name}
                    title={name}
                    data={{ predicted_points, mae: '-', rmse: '-' }}
                    playerName={playerName}
                  />
                ))}
            </div>
          </>
        )}
      </div>
    </>
  );
}




// import React, { useEffect, useState } from 'react';
// import ResultBox from './ResultBox';


// export default function Predictions({ indiv, global, blended, playerName }) {
//   const [globalResult, setGlobalResult] = useState(global);
//   const [individualResult, setIndividualResult] = useState(indiv);
//   const [blendedResult, setBlendedResult] = useState(blended);
//   return (
//     <>
//       {global && (
//         <>
//           <h3 className="text-xl font-medium mb-4 text-center">Global Model</h3>
//           <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//             {Object.entries(global).map(([name, data]) => (
//               <ResultBox
//                 key={name}
//                 title={name}
//                 data={data}
//                 playerName={playerName}
//               />
//             ))}
//           </div>
//         </>
//       )}

//       {indiv && (
//         <>
//           <h3 className="text-xl font-medium mb-4 text-center">Individual Model</h3>
//           <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//             {Object.entries(indiv).map(([name, data]) => (
//               <ResultBox
//                 key={name}
//                 title={name}
//                 data={data}
//                 playerName={playerName}
//               />
//             ))}
//           </div>
//         </>
//       )}

//     {blended && (
//         <>
//           <h3 className="text-xl font-medium mb-4 text-center">Blended Model</h3>
//           <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//             {Object.entries(blended)
//               .filter(([key]) => key !== 'alpha')
//               .map(([name, predicted_points]) => (
//                 <ResultBox
//                   key={name}
//                   title={name}
//                   data={{ predicted_points, mae: '-', rmse: '-' }}
//                   playerName={playerName}
//                 />
//               ))}
//           </div>
//         </>
//       )}
//     </>
//   );
// }