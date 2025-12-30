import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import Controls from './components/Controls'
import StatusCard from './components/StatusCard'
import AccuracyChart from './components/AccuracyChart'
import VersionsTable from './components/VersionsTable'
import RollbackModal from './components/RollbackModal'
import ModelDetailsModal from './components/ModelDetailsModal'
import ProvenanceModal from './components/ProvenanceModal'
import { apiCall } from './utils/api'

const API_BASE = '/api/v1'

function App() {
  const [apiKey, setApiKey] = useState(() => {
    const stored = localStorage.getItem('apiKey')
    return stored || prompt('Enter API Key:', 'your-api-key-here') || 'your-api-key-here'
  })
  const [status, setStatus] = useState(null)
  const [versions, setVersions] = useState([])
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [showRollbackModal, setShowRollbackModal] = useState(false)
  const [showModelDetailsModal, setShowModelDetailsModal] = useState(false)
  const [showProvenanceModal, setShowProvenanceModal] = useState(false)

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('apiKey', apiKey)
      refreshStatus()
      loadVersions()
      const interval = setInterval(() => {
        refreshStatus()
        loadVersions()
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [apiKey])

  const refreshStatus = async () => {
    try {
      const data = await apiCall(API_BASE, '/training/status', 'GET', null, apiKey)
      setStatus(data)
    } catch (err) {
      // Silently fail - status might not be available yet
    }
  }

  const loadVersions = async () => {
    try {
      const data = await apiCall(API_BASE, '/models?limit=50', 'GET', null, apiKey)
      if (data.versions && data.versions.length > 0) {
        setVersions(data.versions)
      }
    } catch (err) {
      // Silently fail - endpoint might not be implemented yet
    }
  }

  const startTraining = async () => {
    try {
      const result = await apiCall(API_BASE, '/training/start', 'POST', {}, apiKey)
      setSuccess(`Training started! Iteration: ${result.iteration}`)
      refreshStatus()
    } catch (err) {
      setError(err.message)
    }
  }

  const stopTraining = async () => {
    try {
      await apiCall(API_BASE, '/training/stop', 'POST', {}, apiKey)
      setSuccess('Training stop requested')
      refreshStatus()
    } catch (err) {
      setError(err.message)
    }
  }

  const listModels = async () => {
    try {
      const data = await apiCall(API_BASE, '/models?limit=100', 'GET', null, apiKey)
      if (data.versions && data.versions.length > 0) {
        setVersions(data.versions)
        setSuccess(`Loaded ${data.versions.length} model versions`)
      } else {
        setError('No model versions found. Start training to create versions.')
      }
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [error])

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [success])

  return (
    <div className="max-w-7xl mx-auto">
      <Header />
      
      {error && (
        <div className="error">
          Error: {error}
        </div>
      )}
      
      {success && (
        <div className="success-message">
          {success}
        </div>
      )}

      <Controls
        isTraining={status?.is_training || status?.status === 'running'}
        onStartTraining={startTraining}
        onStopTraining={stopTraining}
        onRefresh={refreshStatus}
        onRollback={() => setShowRollbackModal(true)}
        onListModels={listModels}
        onViewModelDetails={() => setShowModelDetailsModal(true)}
        onViewProvenance={() => setShowProvenanceModal(true)}
      />

      <StatusCard status={status} />

      <AccuracyChart accuracyHistory={status?.accuracy_history} />

      <VersionsTable versions={versions} onViewDetails={(versionId) => {
        setShowModelDetailsModal(true)
        // Store versionId for modal
        localStorage.setItem('selectedVersionId', versionId)
      }} />

      <RollbackModal
        isOpen={showRollbackModal}
        onClose={() => setShowRollbackModal(false)}
        onSuccess={() => {
          setShowRollbackModal(false)
          refreshStatus()
        }}
        apiKey={apiKey}
        apiBase={API_BASE}
      />

      <ModelDetailsModal
        isOpen={showModelDetailsModal}
        onClose={() => {
          setShowModelDetailsModal(false)
          localStorage.removeItem('selectedVersionId')
        }}
        apiKey={apiKey}
        apiBase={API_BASE}
      />

      <ProvenanceModal
        isOpen={showProvenanceModal}
        onClose={() => setShowProvenanceModal(false)}
        apiKey={apiKey}
        apiBase={API_BASE}
      />
    </div>
  )
}

export default App

