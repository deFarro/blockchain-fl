import React from 'react'

function Controls({
  activeTab,
  onTabChange,
  isTraining,
  onStartTraining,
  onStopTraining,
  onRefresh,
  onListModels,
}) {
  return (
    <div className="bg-white rounded-lg my-5 shadow-md">
      <div className="mb-5 flex gap-2">
          <button
            id="start-training-btn"
            onClick={onStartTraining}
            disabled={isTraining}
            className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base transition-colors hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            ▶ Start Training
          </button>
          <button
            id="stop-training-btn"
            onClick={onStopTraining}
            disabled={!isTraining}
            className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base transition-colors hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            ⏹ Stop Training
          </button>
          <button
            onClick={onRefresh}
            className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base transition-colors hover:bg-indigo-600"
          >
            🔄 Refresh Status
          </button>
          <button
            onClick={() => window.open('/docs', '_blank')}
            className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base transition-colors hover:bg-indigo-600"
          >
            📚 API Docs
          </button>
      </div>

      <div className="border-b border-gray-200">
        <div className="flex gap-2">
          <button
            onClick={() => onTabChange('dashboard')}
            className={`px-6 py-4 text-base font-medium transition-colors border-b-2 ${
              activeTab === 'dashboard'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📊 Dashboard
          </button>
          <button
            onClick={() => {
              onTabChange('models')
              onListModels()
            }}
            className={`px-6 py-4 text-base font-medium transition-colors border-b-2 ${
              activeTab === 'models'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📋 List Models
          </button>
          <button
            onClick={() => onTabChange('rollback')}
            className={`px-6 py-4 text-base font-medium transition-colors border-b-2 ${
              activeTab === 'rollback'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            ↩ Rollback
          </button>
          <button
            onClick={() => onTabChange('provenance')}
            className={`px-6 py-4 text-base font-medium transition-colors border-b-2 ${
              activeTab === 'provenance'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            🔗 Provenance
          </button>
          <button
            onClick={() => onTabChange('details')}
            className={`px-6 py-4 text-base font-medium transition-colors border-b-2 ${
              activeTab === 'details'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            🔍 Model Details
          </button>
        </div>
      </div>
    </div>
  )
}

export default Controls

