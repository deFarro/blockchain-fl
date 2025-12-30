import React from 'react'

function StatusCard({ status }) {
  if (!status) {
    return (
      <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
        <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Training Status</h2>
        <div className="text-center py-5 text-gray-600">Loading status...</div>
      </div>
    )
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'text-green-600'
      case 'completed':
        return 'text-blue-600'
      case 'stopped':
        return 'text-yellow-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
      <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Training Status</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-5">
        <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
          <label className="block text-gray-600 text-sm mb-1">Status</label>
          <div className={`text-2xl font-bold ${getStatusColor(status.status)}`}>
            {status.status || 'stopped'}
          </div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
          <label className="block text-gray-600 text-sm mb-1">Current Iteration</label>
          <div className="text-2xl font-bold text-indigo-500">
            {status.current_iteration !== null ? status.current_iteration : '-'}
          </div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
          <label className="block text-gray-600 text-sm mb-1">Best Accuracy</label>
          <div className="text-2xl font-bold text-green-600">
            {status.best_accuracy !== null ? `${status.best_accuracy.toFixed(2)}%` : '-'}
          </div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
          <label className="block text-gray-600 text-sm mb-1">Total Iterations</label>
          <div className="text-2xl font-bold text-gray-800">{status.total_iterations || 0}</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
          <label className="block text-gray-600 text-sm mb-1">Rollbacks</label>
          <div className="text-2xl font-bold text-yellow-600">{status.rollback_count || 0}</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
          <label className="block text-gray-600 text-sm mb-1">Start Time</label>
          <div className="text-2xl font-bold text-gray-800">
            {status.start_time ? new Date(status.start_time).toLocaleString() : '-'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatusCard

