import React, { useState } from 'react'
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

function Provenance({ apiKey, apiBase }) {
  const [versionId, setVersionId] = useState('')
  const [provenance, setProvenance] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleView = async () => {
    if (!versionId) {
      setError('Please enter a version ID')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const data = await apiCall(apiBase, `/models/${versionId}/provenance?include_chain=true`, 'GET', null, apiKey)
      setProvenance(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
      <h2 className="text-gray-800 mb-5 text-2xl font-semibold">View Provenance Chain</h2>
      <p className="text-gray-600 mb-5">Enter a model version ID to view its provenance chain</p>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-700">
          Error: {error}
        </div>
      )}

      <div className="mb-4">
        <label htmlFor="provenance-version-id" className="block mb-1 text-gray-800 font-medium">Version ID</label>
        <input
          type="text"
          id="provenance-version-id"
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
          {loading ? 'Loading...' : 'View Provenance'}
        </button>
      </div>
      {provenance && (
        <div className="mt-5 bg-gray-50 p-4 rounded-md">
          <strong>Version ID:</strong> {provenance.version_id}
          <br />
          <strong>Parent Version:</strong> {inferParentVersionId(
            provenance.version_id, 
            provenance.parent_version_id,
            provenance.metadata?.iteration
          ) || 'None (initial)'}
          <br />
          <strong>Hash:</strong> {provenance.hash ? `${provenance.hash.substring(0, 16)}...` : '-'}
          <br />
          {provenance.timestamp && (
            <>
              <strong>Timestamp:</strong> {formatTimestamp(provenance.timestamp)}
              <br />
            </>
          )}
          {provenance.chain && provenance.chain.length > 0 ? (
            <>
              <h4 className="mt-3 mb-2 font-semibold">Provenance Chain:</h4>
              <ul className="list-disc pl-5">
                {provenance.chain.map((v) => (
                  <li key={v.version_id}>
                    {v.version_id} (Iteration: {v.iteration || '-'}, Accuracy:{' '}
                    {v.accuracy ? `${v.accuracy.toFixed(2)}%` : '-'})
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="mt-3">No provenance chain available. This may be the initial version.</p>
          )}
        </div>
      )}
    </div>
  )
}

export default Provenance

