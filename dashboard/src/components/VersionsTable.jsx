import React from 'react'

function VersionsTable({ versions, onViewDetails }) {
  const getStatusBadgeClass = (status) => {
    if (!status) return 'bg-blue-100 text-blue-800'
    switch (status.toLowerCase()) {
      case 'passed':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'pending':
        return 'bg-yellow-100 text-yellow-800'
      default:
        return 'bg-blue-100 text-blue-800'
    }
  }

  if (!versions || versions.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Model Versions</h2>
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Version ID</th>
              <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Iteration</th>
              <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Accuracy</th>
              <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Status</th>
              <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Timestamp</th>
              <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td colSpan="6" className="text-center py-5 text-gray-600">
                No model versions yet. Start training to create versions.
                <br />
                <small>Click "📋 List Models" button to refresh.</small>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Model Versions</h2>
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Version ID</th>
            <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Iteration</th>
            <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Accuracy</th>
            <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Status</th>
            <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Timestamp</th>
            <th className="p-3 text-left border-b border-gray-200 bg-gray-50 font-semibold text-gray-800">Actions</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((v) => (
            <tr key={v.version_id} className="hover:bg-gray-50">
              <td className="p-3 text-left border-b border-gray-200">{v.version_id}</td>
              <td className="p-3 text-left border-b border-gray-200">{v.iteration || '-'}</td>
              <td className="p-3 text-left border-b border-gray-200">{v.accuracy ? `${v.accuracy.toFixed(2)}%` : '-'}</td>
              <td className="p-3 text-left border-b border-gray-200">
                <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${getStatusBadgeClass(v.validation_status)}`}>
                  {v.validation_status || 'pending'}
                </span>
              </td>
              <td className="p-3 text-left border-b border-gray-200">{v.timestamp ? new Date(v.timestamp).toLocaleString() : '-'}</td>
              <td className="p-3 text-left border-b border-gray-200">
                <button
                  onClick={() => onViewDetails(v.version_id)}
                  className="px-2 py-1 text-xs bg-indigo-500 text-white rounded hover:bg-indigo-600"
                >
                  View
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default VersionsTable

