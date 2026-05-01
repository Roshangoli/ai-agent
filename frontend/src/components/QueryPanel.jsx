import { useState } from 'react'
import { Send, Sparkles, Lightbulb, Upload, Database, X } from 'lucide-react'
import { submitQuery, uploadCSVForQuery } from '../services/api'

const QueryPanel = ({ onQuerySubmit, loading, setLoading, onStatsUpdate }) => {
  const [question, setQuestion] = useState('')
  const [error, setError] = useState(null)
  const [csvFile, setCsvFile] = useState(null)
  const [uploadedDB, setUploadedDB] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [generatedQuestions, setGeneratedQuestions] = useState(null)

  // Default questions (fallback when no CSV uploaded)
  const defaultQuestions = [
    'Show total sales by region',
    'Which product has the highest sales?',
    'Show monthly sales trends',
    'What are the sales in the East region?',
    'Show top 5 products by revenue',
  ]

  // Use generated questions if available, otherwise use default
  const exampleQuestions = generatedQuestions || defaultQuestions

  const handleCSVUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file')
      return
    }

    setUploading(true)
    setError(null)

    try {
      const result = await uploadCSVForQuery(file)
      setUploadedDB(result.db_path)
      setCsvFile(file)
      setError(null)

      // Store auto-generated questions if available
      if (result.example_questions && result.example_questions.length > 0) {
        setGeneratedQuestions(result.example_questions)
        console.log('✅ Received auto-generated questions:', result.example_questions)
      }

      // Pass dynamic stats to parent component
      if (result.dynamic_stats && onStatsUpdate) {
        onStatsUpdate(result.dynamic_stats)
        console.log('✅ Received dynamic stats:', result.dynamic_stats)
      }
    } catch (err) {
      setError(err.message)
      setCsvFile(null)
      setUploadedDB(null)
      setGeneratedQuestions(null)
    } finally {
      setUploading(false)
    }
  }

  const handleRemoveCSV = () => {
    setCsvFile(null)
    setUploadedDB(null)
    setGeneratedQuestions(null)  // Reset to default questions

    // Reset stats to default
    if (onStatsUpdate) {
      onStatsUpdate(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    setError(null)

    try {
      const result = await submitQuery(question, true, uploadedDB)
      onQuerySubmit(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (example) => {
    setQuestion(example)
  }

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 shadow-2xl">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-yellow-400" />
        <h2 className="text-xl font-semibold text-white">Ask a Question</h2>
      </div>

      {/* CSV Upload Section */}
      <div className="mb-4">
        {!csvFile ? (
          <div>
            <input
              type="file"
              accept=".csv"
              onChange={handleCSVUpload}
              className="hidden"
              id="csv-upload"
              disabled={loading || uploading}
            />
            <label
              htmlFor="csv-upload"
              className={`flex items-center justify-center gap-2 w-full px-4 py-3 border-2 border-dashed rounded-lg cursor-pointer transition-all
                ${uploading ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-blue-500 bg-gray-800/30'}
                ${loading ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                  <span className="text-sm text-blue-300">Analyzing & generating questions...</span>
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-300">Upload your own CSV (optional)</span>
                </>
              )}
            </label>
            <p className="text-xs text-gray-400 mt-1">
              Or use the default sales database
            </p>
          </div>
        ) : (
          <div className="flex items-center justify-between px-4 py-3 bg-green-500/10 border border-green-500/30 rounded-lg">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-300">{csvFile.name}</span>
            </div>
            <button
              onClick={handleRemoveCSV}
              className="text-gray-400 hover:text-red-400 transition-colors"
              disabled={loading}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., Show me total sales by region for last quarter..."
            className="w-full h-32 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            disabled={loading}
          />
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-3 text-red-200 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
        >
          {loading ? (
            <>
              <div className="spinner w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              <span>Analyze</span>
            </>
          )}
        </button>
      </form>

      {/* Example Questions */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-400" />
            <h3 className="text-sm font-medium text-gray-300">
              Example Questions:
            </h3>
          </div>
          {generatedQuestions && (
            <span className="text-xs px-2 py-1 bg-green-500/20 text-green-300 rounded-full border border-green-500/30">
              ✨ AI-Generated
            </span>
          )}
        </div>
        <div className="space-y-2">
          {exampleQuestions.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              className="w-full text-left px-3 py-2 bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700 rounded-lg text-sm text-gray-300 transition-colors"
              disabled={loading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
        <p className="text-sm text-blue-200">
          <strong>💡 Pro Tip:</strong> Ask questions in natural language. The
          AI will automatically generate SQL, execute it, and create
          visualizations!
        </p>
      </div>
    </div>
  )
}

export default QueryPanel