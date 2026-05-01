import { useState, useRef } from 'react'
import { Upload, Download, AlertCircle, CheckCircle, Loader2, FileText, BarChart3 } from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function BatchPrediction({ modelId, taskType, onPredictionComplete }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  // Handle drag events
  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const droppedFiles = e.dataTransfer.files
    if (droppedFiles.length > 0) {
      handleFileSelection(droppedFiles[0])
    }
  }

  // Handle file selection
  const handleFileSelection = (selectedFile) => {
    // Validate file type
    if (!selectedFile.name.endsWith('.csv')) {
      setError('Only CSV files are supported')
      return
    }

    setFile(selectedFile)
    setError(null)
    setResults(null)
  }

  const handleFileInputChange = (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0])
    }
  }

  // Clear file
  const handleClearFile = () => {
    setFile(null)
    setError(null)
    setResults(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Upload and predict
  const handlePredict = async () => {
    if (!file) {
      setError('Please select a CSV file')
      return
    }

    setUploading(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('model_id', modelId)
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/api/datascience/predict-batch`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Batch prediction failed')
      }

      const result = await response.json()
      setResults(result)

      if (onPredictionComplete) {
        onPredictionComplete(result)
      }
    } catch (err) {
      setError(err.message || 'Failed to process batch predictions')
    } finally {
      setUploading(false)
    }
  }

  // Download results
  const handleDownload = () => {
    if (results && results.download_url) {
      const downloadUrl = `${API_BASE_URL}${results.download_url}`
      window.open(downloadUrl, '_blank')
    }
  }

  // Get risk color
  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return 'text-red-400'
      case 'medium':
        return 'text-yellow-400'
      case 'low':
        return 'text-green-400'
      default:
        return 'text-gray-400'
    }
  }

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return '🔴'
      case 'medium':
        return '🟡'
      case 'low':
        return '🟢'
      default:
        return '⚪'
    }
  }

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg shadow-xl p-6 border border-gray-700">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-700">
        <Upload className="w-6 h-6 text-purple-400" />
        <div>
          <h3 className="text-xl font-bold text-white">Batch Predictions</h3>
          <p className="text-sm text-gray-400">Upload CSV file for bulk predictions</p>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-4 bg-red-900/30 border border-red-500/50 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-red-200 text-sm font-semibold">Error</p>
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* File Upload Area */}
      {!results && (
        <div className="space-y-4">
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer ${
              isDragging
                ? 'border-purple-500 bg-purple-900/20'
                : 'border-gray-600 hover:border-gray-500 bg-gray-800/50'
            }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileInputChange}
              className="hidden"
            />

            <Upload className={`w-16 h-16 mx-auto mb-4 ${isDragging ? 'text-purple-400' : 'text-gray-500'}`} />

            {!file ? (
              <>
                <h4 className="text-lg font-semibold text-white mb-2">
                  {isDragging ? 'Drop CSV file here' : 'Drag & drop CSV file'}
                </h4>
                <p className="text-sm text-gray-400">or click to browse</p>
                <p className="text-xs text-gray-500 mt-2">CSV files only • No target column needed</p>
              </>
            ) : (
              <div className="flex items-center justify-center gap-3">
                <FileText className="w-8 h-8 text-green-400" />
                <div className="text-left">
                  <p className="text-white font-semibold">{file.name}</p>
                  <p className="text-sm text-gray-400">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
            )}
          </div>

          {file && (
            <div className="flex gap-3">
              <button
                onClick={handlePredict}
                disabled={uploading}
                className="flex-1 py-3 px-6 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg text-white font-semibold flex items-center justify-center gap-2 transition-all"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    Predict All
                  </>
                )}
              </button>

              <button
                onClick={handleClearFile}
                disabled={uploading}
                className="py-3 px-6 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 rounded-lg text-white font-semibold transition-all"
              >
                Clear
              </button>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-6">
          {/* Success Message */}
          <div className="p-4 bg-green-900/30 border border-green-500/50 rounded-lg flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <div>
              <p className="text-green-200 font-semibold">Predictions Complete!</p>
              <p className="text-green-300 text-sm">{results.total_rows} rows processed</p>
            </div>
          </div>

          {/* Summary */}
          {taskType === 'classification' && results.summary && (
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-400" />
                Prediction Summary
              </h4>

              <div className="grid grid-cols-3 gap-4">
                <div className="bg-red-900/30 border border-red-500/30 rounded-lg p-4">
                  <div className="text-sm text-red-300 mb-1">High Risk</div>
                  <div className="text-3xl font-bold text-red-400">
                    {results.summary.high_risk || 0}
                  </div>
                  <div className="text-xs text-red-400/70 mt-1">
                    {((results.summary.high_risk || 0) / results.total_rows * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="bg-yellow-900/30 border border-yellow-500/30 rounded-lg p-4">
                  <div className="text-sm text-yellow-300 mb-1">Medium Risk</div>
                  <div className="text-3xl font-bold text-yellow-400">
                    {results.summary.medium_risk || 0}
                  </div>
                  <div className="text-xs text-yellow-400/70 mt-1">
                    {((results.summary.medium_risk || 0) / results.total_rows * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4">
                  <div className="text-sm text-green-300 mb-1">Low Risk</div>
                  <div className="text-3xl font-bold text-green-400">
                    {results.summary.low_risk || 0}
                  </div>
                  <div className="text-xs text-green-400/70 mt-1">
                    {((results.summary.low_risk || 0) / results.total_rows * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Sample Predictions */}
          {results.sample_predictions && results.sample_predictions.length > 0 && (
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-4">Sample Predictions (first 20 rows)</h4>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left text-gray-400 font-semibold py-2 px-3">Row</th>
                      <th className="text-left text-gray-400 font-semibold py-2 px-3">Prediction</th>
                      {results.sample_predictions[0]?.probability !== undefined && (
                        <th className="text-left text-gray-400 font-semibold py-2 px-3">Probability</th>
                      )}
                      {results.sample_predictions[0]?.risk_level && (
                        <th className="text-left text-gray-400 font-semibold py-2 px-3">Risk</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {results.sample_predictions.map((pred, idx) => (
                      <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                        <td className="py-2 px-3 text-gray-300">{pred.row}</td>
                        <td className="py-2 px-3 text-white font-semibold">
                          {typeof pred.prediction === 'number' && taskType === 'regression'
                            ? pred.prediction.toFixed(2)
                            : pred.prediction}
                        </td>
                        {pred.probability !== undefined && (
                          <td className="py-2 px-3 text-white">
                            {(pred.probability * 100).toFixed(1)}%
                          </td>
                        )}
                        {pred.risk_level && (
                          <td className="py-2 px-3">
                            <span className={`flex items-center gap-1 ${getRiskColor(pred.risk_level)}`}>
                              <span>{getRiskIcon(pred.risk_level)}</span>
                              <span className="capitalize">{pred.risk_level}</span>
                            </span>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Download Button */}
          <div className="flex gap-3">
            <button
              onClick={handleDownload}
              className="flex-1 py-3 px-6 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 rounded-lg text-white font-semibold flex items-center justify-center gap-2 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Full Results (CSV)
            </button>

            <button
              onClick={() => { handleClearFile(); setResults(null); }}
              className="py-3 px-6 bg-gray-700 hover:bg-gray-600 rounded-lg text-white font-semibold transition-all"
            >
              New Prediction
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
