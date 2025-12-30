import React from 'react'

function Controls({
  isTraining,
  onStartTraining,
  onStopTraining,
  onRefresh,
  onRollback,
  onListModels,
  onViewModelDetails,
  onViewProvenance,
}) {
  return (
    <div className="bg-white p-5 rounded-lg mb-5 shadow-md">
      <button
        id="start-training-btn"
        onClick={onStartTraining}
        disabled={isTraining}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
      >
        ▶ Start Training
      </button>
      <button
        id="stop-training-btn"
        onClick={onStopTraining}
        disabled={!isTraining}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
      >
        ⏹ Stop Training
      </button>
      <button
        onClick={onRefresh}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600"
      >
        🔄 Refresh Status
      </button>
      <button
        onClick={onRollback}
        className="bg-red-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-red-600"
      >
        ↩ Rollback to Version
      </button>
      <button
        onClick={onListModels}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600"
      >
        📋 List Models
      </button>
      <button
        onClick={onViewModelDetails}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600"
      >
        🔍 View Model Details
      </button>
      <button
        onClick={onViewProvenance}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600"
      >
        🔗 View Provenance Chain
      </button>
      <button
        onClick={() => window.open('/docs', '_blank')}
        className="bg-indigo-500 text-white border-none py-3 px-6 rounded-md cursor-pointer text-base mr-2.5 mb-2.5 transition-colors hover:bg-indigo-600"
      >
        📚 API Docs
      </button>
    </div>
  )
}

export default Controls

