import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import Controls from './components/Controls'
import StatusCard from './components/StatusCard'
import VersionsTable from './components/VersionsTable'
import Rollback from './components/RollbackModal'
import ModelDetails from './components/ModelDetailsModal'
import Provenance from './components/ProvenanceModal'
import { apiCall } from './utils/api'

const API_BASE = '/api/v1'

function App() {
  const [apiKey, setApiKey] = useState(() => {
    const stored = localStorage.getItem('apiKey')
    return stored || prompt('Enter API Key:', 'your-api-key-here') || 'your-api-key-here'
  })
  const [activeTab, setActiveTab] = useState('dashboard')
  const [status, setStatus] = useState(null)
  const [versions, setVersions] = useState([])
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [selectedVersionId, setSelectedVersionId] = useState(null)
  const [isLoadingStatus, setIsLoadingStatus] = useState(false)
  const [isLoadingVersions, setIsLoadingVersions] = useState(false)

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('apiKey', apiKey)
      setTimeout(() => {
        refreshStatus()
        loadVersions()
      }, 0)
      const interval = setInterval(() => {
        refreshStatus()
        loadVersions()
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [apiKey])

  const refreshStatus = async () => {
    setIsLoadingStatus(true)
    try {
      const data = await apiCall(API_BASE, '/training/status', 'GET', null, apiKey)
      setStatus(data)
    } catch (err) {
      // Silently fail - status might not be available yet
    } finally {
      setIsLoadingStatus(false)
    }
  }

  const loadVersions = async () => {
    setIsLoadingVersions(true)
    try {
      const data = await apiCall(API_BASE, '/models?limit=50', 'GET', null, apiKey)
      if (data.versions && data.versions.length > 0) {
        setVersions(data.versions)
      }
    } catch (err) {
      // Silently fail - endpoint might not be implemented yet
    } finally {
      setIsLoadingVersions(false)
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
      setActiveTab('models')
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
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isTraining={status?.is_training || status?.status === 'running'}
        onStartTraining={startTraining}
        onStopTraining={stopTraining}
        onRefresh={refreshStatus}
        onListModels={listModels}
      />

      {activeTab === 'dashboard' && (
        <StatusCard status={status} isLoading={isLoadingStatus} />
      )}

      {activeTab === 'models' && (
        <VersionsTable versions={versions} isLoading={isLoadingVersions} onViewDetails={(versionId) => {
          setSelectedVersionId(versionId)
          setActiveTab('details')
        }} />
      )}

      {activeTab === 'rollback' && (
        <Rollback
          onSuccess={() => {
            refreshStatus()
            setSuccess('Rollback successful!')
          }}
          apiKey={apiKey}
          apiBase={API_BASE}
        />
      )}

      {activeTab === 'provenance' && (
        <Provenance
          apiKey={apiKey}
          apiBase={API_BASE}
        />
      )}

      {activeTab === 'details' && (
        <ModelDetails
          apiKey={apiKey}
          apiBase={API_BASE}
          initialVersionId={selectedVersionId}
        />
      )}
    </div>
  )
}

export default App

