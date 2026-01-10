import React, { useState } from 'react'
import { apiCall } from '../utils/api'

function Rollback({ onSuccess, apiKey, apiBase }) {
  const [versionId, setVersionId] = useState('')
  const [reason, setReason] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!versionId || !reason) {
      setError('Please fill in all fields')
      return
    }

    setLoading(true)
    setError(null)
    setSuccess(null)
    try {
      await apiCall(apiBase, `/models/${versionId}/rollback`, 'POST', { target_version_id: versionId, reason }, apiKey)
      setSuccess('Rollback successful!')
      setVersionId('')
      setReason('')
      if (onSuccess) onSuccess()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white p-6 rounded-lg mb-5 shadow-md">
      <h2 className="text-gray-800 mb-5 text-2xl font-semibold">Rollback to Version</h2>
      <p className="text-gray-600 mb-5">Select a model version to rollback to</p>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-700">
          Error: {error}
        </div>
      )}
      
      {success && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md text-green-700">
          {success}
        </div>
      )}

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
        </div>
      </form>
    </div>
  )
}

export default Rollback

