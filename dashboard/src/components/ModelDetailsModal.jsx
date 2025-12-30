import React, { useState, useEffect } from 'react'
import { apiCall } from '../utils/api'

function ModelDetailsModal({ isOpen, onClose, apiKey, apiBase }) {
  const [versionId, setVersionId] = useState('')
  const [model, setModel] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleViewWithId = async (id) => {
    setLoading(true)
    try {
      const data = await apiCall(apiBase, `/models/${id}`, 'GET', null, apiKey)
      setModel(data)
    } catch (error) {
      alert(`Error: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      const stored = localStorage.getItem('selectedVersionId')
      if (stored) {
        setVersionId(stored)
        localStorage.removeItem('selectedVersionId')
        // Auto-load if version ID was set
        handleViewWithId(stored)
      }
    } else {
      setVersionId('')
      setModel(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen])

  if (!isOpen) return null

  const handleView = async () => {
    if (!versionId) {
      alert('Please enter a version ID')
      return
    }
    await handleViewWithId(versionId)
  }

  return (
    <div className="fixed top-0 left-0 w-full h-full bg-black bg-opacity-50 z-[1000] flex items-center justify-center" onClick={onClose}>
      <div className="bg-white p-8 rounded-lg max-w-md w-[90%] max-h-[90vh] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-5">
          <h3 className="text-gray-800 mb-2.5 text-xl font-semibold">View Model Details</h3>
          <p className="text-gray-600">Enter a model version ID to view details</p>
        </div>
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
        <div className="flex gap-2.5 mt-5">
          <button
            onClick={handleView}
            disabled={loading}
            className="bg-indigo-500 text-white border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Loading...' : 'View Details'}
          </button>
          <button
            onClick={onClose}
            className="bg-gray-300 text-gray-800 border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-gray-400"
          >
            Cancel
          </button>
        </div>
        {model && (
          <div className="mt-5 bg-gray-50 p-4 rounded-md font-mono text-xs">
            <strong>Version ID:</strong> {model.version_id}
            <br />
            <strong>Parent Version:</strong> {model.parent_version_id || 'None (initial)'}
            <br />
            <strong>Iteration:</strong> {model.iteration || '-'}
            <br />
            <strong>Accuracy:</strong> {model.accuracy ? `${model.accuracy.toFixed(2)}%` : '-'}
            <br />
            <strong>IPFS CID:</strong> {model.ipfs_cid || '-'}
            <br />
            <strong>Hash:</strong> {model.hash ? `${model.hash.substring(0, 16)}...` : '-'}
            <br />
            <strong>Timestamp:</strong> {model.timestamp ? new Date(model.timestamp).toLocaleString() : '-'}
            <br />
            <strong>Num Clients:</strong> {model.num_clients || '-'}
            <br />
            <strong>Client IDs:</strong> {model.client_ids ? model.client_ids.join(', ') : '-'}
          </div>
        )}
      </div>
    </div>
  )
}

export default ModelDetailsModal

