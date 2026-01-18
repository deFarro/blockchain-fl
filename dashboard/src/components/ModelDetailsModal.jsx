import React, { useState, useEffect } from 'react'
import { apiCall } from '../utils/api'
import { formatTimestamp } from '../utils/format'

// Helper function to infer parent version ID from version ID pattern
function inferParentVersionId(versionId, parentVersionId, iteration) {
  // If parent_version_id exists and is not empty, use it
  if (parentVersionId && parentVersionId.trim() !== '') {
    return parentVersionId
  }
  
  // If iteration is provided and > 0, parent should exist
  if (iteration !== null && iteration !== undefined && iteration > 0) {
    return `Missing (should be model_v${iteration - 1}_...)`
  }
  
  // Try to infer from version ID pattern: model_v{iteration}_{timestamp}_{unique_id}
  const match = versionId.match(/^model_v(\d+)_/)
  if (match) {
    const iter = parseInt(match[1], 10)
    if (iter > 0) {
      // For iteration > 0, parent should be iteration - 1
      return `Missing (should be model_v${iter - 1}_...)`
    }
  }
  
  return null
}

function ModelDetails({ apiKey, apiBase, initialVersionId }) {
  const [versionId, setVersionId] = useState('')
  const [model, setModel] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleViewWithId = async (id) => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiCall(apiBase, `/models/${id}`, 'GET', null, apiKey)
      setModel(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (initialVersionId) {
      setVersionId(initialVersionId)
      handleViewWithId(initialVersionId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialVersionId])

  const handleView = async () => {
    if (!versionId) {
      setError('Please enter a version ID')
      return
    }
    await handleViewWithId(versionId)
  }

  return (
    <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
      <h2 className="text-gray-800 mb-5 text-2xl font-semibold">View Model Details</h2>
      <p className="text-gray-600 mb-5">Enter a model version ID to view details</p>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-700">
          Error: {error}
        </div>
      )}

      <div className="mb-4">
        <label htmlFor="model-version-id" className="block mb-1 text-gray-800 font-medium">Version ID</label>
        <input
          type="text"
          id="model-version-id"
          value={versionId}
          onChange={(e) => setVersionId(e.target.value)}
          placeholder="e.g., version_5"
          className="w-full p-2.5 border border-gray-300 rounded-md text-sm"
        />
      </div>
      <div className="flex gap-2.5 mt-5 mb-5">
        <button
          onClick={handleView}
          disabled={loading}
          className="bg-indigo-500 text-white border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {loading ? 'Loading...' : 'View Details'}
        </button>
      </div>
      {model && (
        <div className="mt-5 bg-gray-50 p-4 rounded-md font-mono text-xs">
          <strong>Version ID:</strong> {model.version_id}
          <br />
          <strong>Parent Version:</strong> {inferParentVersionId(model.version_id, model.parent_version_id, model.iteration) || 'None (initial)'}
          <br />
          <strong>Iteration:</strong> {model.iteration || '-'}
          <br />
          <strong>Accuracy:</strong> {model.accuracy ? `${model.accuracy.toFixed(2)}%` : '-'}
          <br />
          <strong>IPFS CID:</strong> {model.ipfs_cid || '-'}
          <br />
          <strong>Hash:</strong> {model.hash ? `${model.hash.substring(0, 16)}...` : '-'}
          <br />
          <strong>Timestamp:</strong> {formatTimestamp(model.timestamp)}
          <br />
          <strong>Num Clients:</strong> {model.num_clients || '-'}
          <br />
          <strong>Client IDs:</strong> {model.client_ids ? model.client_ids.join(', ') : '-'}
        </div>
      )}
    </div>
  )
}

export default ModelDetails

