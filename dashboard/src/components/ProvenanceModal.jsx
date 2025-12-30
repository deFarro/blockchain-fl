import React, { useState } from 'react'
import { apiCall } from '../utils/api'

function ProvenanceModal({ isOpen, onClose, apiKey, apiBase }) {
  const [versionId, setVersionId] = useState('')
  const [provenance, setProvenance] = useState(null)
  const [loading, setLoading] = useState(false)

  if (!isOpen) return null

  const handleView = async () => {
    if (!versionId) {
      alert('Please enter a version ID')
      return
    }

    setLoading(true)
    try {
      const data = await apiCall(apiBase, `/models/${versionId}/provenance?include_chain=true`, 'GET', null, apiKey)
      setProvenance(data)
    } catch (error) {
      alert(`Error: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed top-0 left-0 w-full h-full bg-black bg-opacity-50 z-[1000] flex items-center justify-center" onClick={onClose}>
      <div className="bg-white p-8 rounded-lg max-w-md w-[90%] max-h-[90vh] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-5">
          <h3 className="text-gray-800 mb-2.5 text-xl font-semibold">View Provenance Chain</h3>
          <p className="text-gray-600">Enter a model version ID to view its provenance chain</p>
        </div>
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
        <div className="flex gap-2.5 mt-5">
          <button
            onClick={handleView}
            disabled={loading}
            className="bg-indigo-500 text-white border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Loading...' : 'View Provenance'}
          </button>
          <button
            onClick={onClose}
            className="bg-gray-300 text-gray-800 border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-gray-400"
          >
            Cancel
          </button>
        </div>
        {provenance && (
          <div className="mt-5 bg-gray-50 p-4 rounded-md">
            <strong>Version ID:</strong> {provenance.version_id}
            <br />
            <strong>Parent Version:</strong> {provenance.parent_version_id || 'None (initial)'}
            <br />
            <strong>Hash:</strong> {provenance.hash ? `${provenance.hash.substring(0, 16)}...` : '-'}
            <br />
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
    </div>
  )
}

export default ProvenanceModal

