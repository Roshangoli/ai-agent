import { useState } from 'react'
import { Upload, Brain, Sparkles, AlertCircle } from 'lucide-react'

export default function DataSciencePanel({ onPipelineStart, loading }) {
  const [file, setFile] = useState(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [taskType, setTaskType] = useState('auto')
  const [columns, setColumns] = useState([])
  const [error, setError] = useState('')

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0]
    if (!selectedFile) return

    if (!selectedFile.name.endsWith('.csv')) {
      setError('Please upload a CSV file')
      return
    }

    setFile(selectedFile)
    setError('')

    // Parse CSV to get column names
    const text = await selectedFile.text()
    const firstLine = text.split('\n')[0]
    const cols = firstLine.split(',').map(col => col.trim())
    setColumns(cols)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!file) {
      setError('Please select a CSV file')
      return
    }

    if (!targetColumn) {
      setError('Please select a target column')
      return
    }

    setError('')
    onPipelineStart({ file, targetColumn, taskType })
  }

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg shadow-xl p-6 border border-gray-700">
      <div className="flex items-center gap-2 mb-6">
        <Brain className="w-6 h-6 text-purple-400" />
        <h2 className="text-xl font-bold text-white">Data Science Mode</h2>
        <span className="ml-auto text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded">
          Autonomous ML Pipeline
        </span>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Upload Dataset (CSV)
          </label>
          <div className="relative">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
              disabled={loading}
            />
            <label
              htmlFor="file-upload"
              className={`flex items-center justify-center gap-2 w-full px-4 py-8 border-2 border-dashed rounded-lg cursor-pointer transition-all
                ${file
                  ? 'border-green-500 bg-green-500/10'
                  : 'border-gray-600 hover:border-purple-500 bg-gray-800/50'
                }
                ${loading ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              <Upload className={`w-6 h-6 ${file ? 'text-green-400' : 'text-gray-400'}`} />
              <span className={file ? 'text-green-300' : 'text-gray-400'}>
                {file ? file.name : 'Click to upload CSV file'}
              </span>
            </label>
          </div>
          {file && (
            <p className="mt-2 text-sm text-green-400">
              ✓ {columns.length} columns detected
            </p>
          )}
        </div>

        {/* Target Column */}
        {columns.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Target Column (to predict)
            </label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={loading}
            >
              <option value="">Select target column...</option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Task Type */}
        {columns.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Task Type
            </label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={loading}
            >
              <option value="auto">Auto-detect (AI decides)</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !file || !targetColumn}
          className={`w-full py-4 px-6 rounded-lg font-semibold text-white flex items-center justify-center gap-2 transition-all
            ${loading || !file || !targetColumn
              ? 'bg-gray-700 cursor-not-allowed'
              : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-lg hover:shadow-purple-500/50'
            }
          `}
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Training ML Model...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Run Autonomous ML Pipeline</span>
            </>
          )}
        </button>

        {/* Info Box */}
        <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <h3 className="text-sm font-semibold text-blue-300 mb-2">
            💡 What will happen:
          </h3>
          <ul className="text-xs text-blue-200 space-y-1">
            <li>✓ 8 AI agents will analyze your data</li>
            <li>✓ Auto clean (missing values, outliers, duplicates)</li>
            <li>✓ Auto feature engineering</li>
            <li>✓ Train multiple ML models</li>
            <li>✓ Pick the best one (85%+ accuracy typical)</li>
            <li>✓ Time: 60-180 seconds | Cost: ~$0.10-$0.20</li>
          </ul>
        </div>

        {/* Example Datasets */}
        <div className="text-xs text-gray-400">
          <p className="font-semibold mb-2">📁 Example datasets to try:</p>
          <ul className="space-y-1">
            <li>• customer_churn_sample.csv (classification)</li>
            <li>• sales_data.csv (regression)</li>
            <li>• Any CSV with 100+ rows</li>
          </ul>
        </div>
      </form>
    </div>
  )
}
