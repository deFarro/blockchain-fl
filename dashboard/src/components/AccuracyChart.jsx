import React from 'react'

function AccuracyChart({ accuracyHistory }) {
  if (!accuracyHistory || accuracyHistory.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
        <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Accuracy History</h2>
        <div className="h-72 relative">
          <div className="text-center py-5 text-gray-600">No accuracy data yet</div>
        </div>
      </div>
    )
  }

  const maxAccuracy = Math.max(...accuracyHistory, 100)
  const minAccuracy = Math.min(...accuracyHistory, 0)
  const range = maxAccuracy - minAccuracy || 1

  return (
    <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
      <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Accuracy History</h2>
      <div className="h-72 relative">
        <div className="flex items-end h-64 gap-2.5 py-5">
          {accuracyHistory.map((acc, idx) => {
            const height = ((acc - minAccuracy) / range) * 100
            return (
              <div key={idx} className="flex-1 flex flex-col items-center min-w-5">
                <div
                  className="w-full bg-gradient-to-t from-indigo-500 to-purple-600 rounded-t transition-all hover:opacity-80 relative"
                  style={{ height: `${height}%` }}
                  title={`Iteration ${idx + 1}: ${acc.toFixed(2)}%`}
                />
                <div className="text-center mt-1 text-xs text-gray-600">{idx + 1}</div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default AccuracyChart

