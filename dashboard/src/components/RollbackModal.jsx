import React, { useState } from 'react'
import { apiCall } from '../utils/api'

function RollbackModal({ isOpen, onClose, onSuccess, apiKey, apiBase }) {
  const [versionId, setVersionId] = useState('')
  const [reason, setReason] = useState('')
  const [loading, setLoading] = useState(false)

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!versionId || !reason) {
      alert('Please fill in all fields')
      return
    }

    setLoading(true)
    try {
      await apiCall(apiBase, `/models/${versionId}/rollback`, 'POST', { target_version_id: versionId, reason }, apiKey)
      onSuccess()
      setVersionId('')
      setReason('')
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
          <h3 className="text-gray-800 mb-2.5 text-xl font-semibold">Rollback to Version</h3>
          <p className="text-gray-600">Select a model version to rollback to</p>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="rollback-version" className="block mb-1 text-gray-800 font-medium">Version ID</label>
            <input
              type="text"
              id="rollback-version"
              value={versionId}
              onChange={(e) => setVersionId(e.target.value)}
              placeholder="e.g., version_5"
              className="w-full p-2.5 border border-gray-300 rounded-md text-sm"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="rollback-reason" className="block mb-1 text-gray-800 font-medium">Reason</label>
            <input
              type="text"
              id="rollback-reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g., Model poisoning detected"
              className="w-full p-2.5 border border-gray-300 rounded-md text-sm"
            />
          </div>
          <div className="flex gap-2.5 mt-5">
            <button
              type="submit"
              className="bg-red-500 text-white border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-red-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              disabled={loading}
            >
              {loading ? 'Processing...' : 'Rollback'}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="bg-gray-300 text-gray-800 border-none py-2.5 px-6 rounded-md cursor-pointer hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default RollbackModal

