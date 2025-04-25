export default function ResultBox({ title, data, playerName }) {
    const format = (val) => val !== undefined ? Number(val).toFixed(2) : '—';
    const formatPlayerName = (name) => name
    .split(/[_-]/)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ');
    return (
      <div className="w-full sm:w-80 bg-[#1e2147] border border-indigo-400 rounded-lg p-4 text-center">
        <h3 className="text-xl font-bold text-indigo-200 mb-3">{title.toUpperCase()}</h3>
        <p className="font-medium text-indigo-100">Predicted Points:</p>
        <div className="text-4xl font-extrabold text-yellow-300 mt-1">{format(data.predicted_points)}</div>
      </div>
    );
  }