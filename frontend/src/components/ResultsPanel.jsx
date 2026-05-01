import { FileText, BarChart3, Database, CheckCircle } from 'lucide-react'

const ResultsPanel = ({ results, loading }) => {
  if (loading) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 shadow-2xl flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="spinner w-12 h-12 border-4 border-white/20 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-300 font-medium">AI Agents Working...</p>
          <p className="text-sm text-gray-400 mt-2">
            Generating SQL → Executing Query → Creating Visualization
          </p>
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 shadow-2xl flex items-center justify-center min-h-[400px]">
        <div className="text-center text-gray-400">
          <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium">No Results Yet</p>
          <p className="text-sm mt-2">
            Ask a question to see AI-powered analytics
          </p>
        </div>
      </div>
    )
  }

  if (!results.success) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 shadow-2xl">
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-200 font-medium">❌ Error</p>
          <p className="text-red-300 text-sm mt-2">
            {results.error || 'Failed to process query'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 shadow-2xl space-y-6">
      {/* Success Header */}
      <div className="flex items-center gap-2">
        <CheckCircle className="w-5 h-5 text-green-400" />
        <h2 className="text-xl font-semibold text-white">Analysis Complete</h2>
      </div>

      {/* Narrative */}
      {results.narrative && (
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <FileText className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-semibold text-blue-300 mb-2">
                AI Insights
              </h3>
              <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">
                {results.narrative}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Chart/Visualization */}
      {results.chart && (
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-5 h-5 text-green-400" />
            <h3 className="text-sm font-semibold text-white">Visualization</h3>
          </div>
          {/* Render base64 image if available */}
          {results.chart.data && results.chart.success ? (
            <div className="bg-gray-800 rounded p-4 flex justify-center">
              <img
                src={results.chart.data}
                alt="Chart Visualization"
                className="max-w-full h-auto rounded shadow-lg"
              />
            </div>
          ) : (
            <div className="bg-gray-800 rounded p-4 text-center text-gray-400">
              <p className="text-sm">Chart generation failed or not available</p>
            </div>
          )}
        </div>
      )}

      {/* Data Table Preview */}
      {results.data && results.data.length > 0 && (
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-5 h-5 text-purple-400" />
            <h3 className="text-sm font-semibold text-white">
              Data Preview ({results.data.length} rows)
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  {Object.keys(results.data[0]).map((key) => (
                    <th
                      key={key}
                      className="text-left px-3 py-2 text-gray-300 font-medium"
                    >
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.data.slice(0, 5).map((row, idx) => (
                  <tr key={idx} className="border-b border-gray-800">
                    {Object.values(row).map((value, vidx) => (
                      <td key={vidx} className="px-3 py-2 text-gray-300">
                        {typeof value === 'number'
                          ? value.toLocaleString()
                          : value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {results.data.length > 5 && (
              <p className="text-xs text-gray-400 mt-2 text-center">
                ... and {results.data.length - 5} more rows
              </p>
            )}
          </div>
        </div>
      )}

      {/* SQL Query (if available) */}
      {results.sql && (
        <details className="bg-gray-900/50 rounded-lg border border-gray-700">
          <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-gray-300 hover:text-white">
            View Generated SQL
          </summary>
          <div className="px-4 pb-3">
            <pre className="text-xs text-green-400 bg-gray-950 p-3 rounded overflow-x-auto">
              {results.sql}
            </pre>
          </div>
        </details>
      )}
    </div>
  )
}

export default ResultsPanel