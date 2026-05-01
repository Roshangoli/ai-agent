import { useState, useEffect } from 'react'
import { BarChart3, Database, Sparkles, Activity, MessageSquare, Brain } from 'lucide-react'
import QueryPanel from './components/QueryPanel'
import ResultsPanel from './components/ResultsPanel'
import DataSciencePanel from './components/DataSciencePanel'
import DataScienceResults from './components/DataScienceResults'
import StatsPanel from './components/StatsPanel'
import Header from './components/Header'
import { getHealth, getStats, runDataScience } from './services/api'
import './App.css'

function App() {
  const [mode, setMode] = useState('query') // 'query' or 'datascience'
  const [results, setResults] = useState(null)
  const [dsResults, setDsResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [dynamicStats, setDynamicStats] = useState(null)  // NEW: For CSV-based stats
  const [health, setHealth] = useState({ status: 'checking' })

  // Check API health on mount
  useEffect(() => {
    checkHealth()
    loadStats()
  }, [])

  const checkHealth = async () => {
    try {
      const healthData = await getHealth()
      setHealth(healthData)
    } catch (error) {
      setHealth({ status: 'error', error: error.message })
    }
  }

  const loadStats = async () => {
    try {
      const statsData = await getStats()
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const handleQuerySubmit = async (queryResults) => {
    setResults(queryResults)
    loadStats()
  }

  const handleDataSciencePipeline = async ({ file, targetColumn, taskType }) => {
    setLoading(true)
    setDsResults(null)

    try {
      const results = await runDataScience(file, targetColumn, taskType)
      setDsResults(results)
      loadStats()
    } catch (error) {
      setDsResults({
        success: false,
        error: error.message || 'Pipeline failed. Please try again.'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleStatsUpdate = (newStats) => {
    setDynamicStats(newStats)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      <Header health={health} />

      <main className="container mx-auto px-4 py-8">
        {/* Tech Stack Badge */}
        <div className="mb-8 flex items-center justify-center gap-2 text-sm text-gray-300">
          <Sparkles className="w-4 h-4 text-yellow-400" />
          <span>Powered by GPT-4 + AutoGen + LangChain</span>
          <span className="mx-2">|</span>
          <Database className="w-4 h-4 text-blue-400" />
          <span>AWS Redshift-Ready</span>
          <span className="mx-2">|</span>
          <BarChart3 className="w-4 h-4 text-green-400" />
          <span>Real-time Analytics</span>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex bg-gray-800 rounded-lg p-1 border border-gray-700">
            <button
              onClick={() => setMode('query')}
              className={`flex items-center gap-2 px-6 py-3 rounded-md font-semibold transition-all ${
                mode === 'query'
                  ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <MessageSquare className="w-5 h-5" />
              Query Mode
            </button>
            <button
              onClick={() => setMode('datascience')}
              className={`flex items-center gap-2 px-6 py-3 rounded-md font-semibold transition-all ${
                mode === 'datascience'
                  ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Brain className="w-5 h-5" />
              Data Science Mode
            </button>
          </div>
        </div>

        {/* Mode Description */}
        <div className="text-center mb-8">
          {mode === 'query' ? (
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Query Mode</h2>
              <p className="text-gray-400">
                Ask questions in natural language. AI generates SQL, executes it, and creates visualizations.
                <span className="text-blue-400 ml-2">⚡ 5-15 seconds</span>
              </p>
            </div>
          ) : (
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Data Science Mode</h2>
              <p className="text-gray-400">
                Upload CSV and train ML models autonomously. 8 AI agents handle everything.
                <span className="text-purple-400 ml-2">🧠 60-180 seconds</span>
              </p>
            </div>
          )}
        </div>

        {/* Stats Panel - Use dynamic stats if available, otherwise default stats */}
        {(dynamicStats || stats) && (
          <StatsPanel stats={dynamicStats || stats} isDynamic={!!dynamicStats} />
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
          {mode === 'query' ? (
            <>
              {/* Query Panel */}
              <QueryPanel
                onQuerySubmit={handleQuerySubmit}
                loading={loading}
                setLoading={setLoading}
                onStatsUpdate={handleStatsUpdate}
              />

              {/* Results Panel */}
              <ResultsPanel results={results} loading={loading} />
            </>
          ) : (
            <>
              {/* Data Science Panel */}
              <DataSciencePanel
                onPipelineStart={handleDataSciencePipeline}
                loading={loading}
              />

              {/* Data Science Results */}
              <DataScienceResults
                results={dsResults}
                loading={loading}
              />
            </>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-400 text-sm">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-green-400" />
            <span>
              AI Multi-Agent Data Analytics System • 90% Faster Processing
            </span>
          </div>
          <p>Built with React + FastAPI + LangChain + AutoGen</p>
        </footer>
      </main>
    </div>
  )
}

export default App
