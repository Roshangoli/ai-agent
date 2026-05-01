import { useState } from 'react'
import { CheckCircle, TrendingUp, Target, Zap, BarChart, Download, Activity, Upload, Package } from 'lucide-react'
import PredictionInterface from './PredictionInterface'
import BatchPrediction from './BatchPrediction'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function DataScienceResults({ results, loading }) {
  const [activeTab, setActiveTab] = useState('results')

  const handleDownloadReport = () => {
    if (results?.report_filename) {
      // Open report in new tab
      const reportUrl = `${API_BASE_URL}/api/datascience/report/${results.report_filename}`
      window.open(reportUrl, '_blank')
    }
  }

  const handleDownloadModel = () => {
    if (results?.model_id) {
      // Download model package
      const modelUrl = `${API_BASE_URL}/api/datascience/download-model/${results.model_id}`
      window.open(modelUrl, '_blank')
    }
  }
  if (loading) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg shadow-xl p-8 border border-gray-700">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-purple-500"></div>
          <h3 className="text-xl font-bold text-white">AI Agents Working...</h3>
          <div className="space-y-2 text-sm text-gray-400 w-full max-w-md">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Step 1: Data Ingestion</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
              <span>Step 2: Autonomous Cleaning</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span>Step 3: Exploratory Analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
              <span>Step 4-8: Feature Engineering → Training → Evaluation</span>
            </div>
          </div>
          <p className="text-xs text-gray-500">This takes 60-180 seconds...</p>
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg shadow-xl p-8 border border-gray-700">
        <div className="text-center text-gray-400">
          <BarChart className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Upload a dataset to get started</p>
          <p className="text-sm mt-2">AI will train ML models autonomously</p>
        </div>
      </div>
    )
  }

  if (!results.success) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg shadow-xl p-6 border border-red-500/50">
        <div className="text-center">
          <div className="text-red-400 mb-4">
            <svg className="w-16 h-16 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-white mb-2">Pipeline Error</h3>
          <p className="text-gray-400">{results.error || 'An unknown error occurred'}</p>
        </div>
      </div>
    )
  }

  const { dataset_info, best_model_name, best_cv_score, best_test_score, observability, model_id, feature_names, model_selection_metadata } = results

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg shadow-xl p-6 border border-gray-700 space-y-6">
      {/* Success Header */}
      <div className="flex items-center gap-3 pb-4 border-b border-gray-700">
        <CheckCircle className="w-8 h-8 text-green-400" />
        <div>
          <h2 className="text-2xl font-bold text-white">Pipeline Complete!</h2>
          <p className="text-sm text-gray-400">ML model trained successfully</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-700">
        <button
          onClick={() => setActiveTab('results')}
          className={`px-4 py-2 font-semibold transition-all flex items-center gap-2 ${
            activeTab === 'results'
              ? 'text-purple-400 border-b-2 border-purple-400'
              : 'text-gray-400 hover:text-gray-300'
          }`}
        >
          <BarChart className="w-4 h-4" />
          Training Results
        </button>
        <button
          onClick={() => setActiveTab('predict')}
          className={`px-4 py-2 font-semibold transition-all flex items-center gap-2 ${
            activeTab === 'predict'
              ? 'text-purple-400 border-b-2 border-purple-400'
              : 'text-gray-400 hover:text-gray-300'
          }`}
          disabled={!model_id || !feature_names}
        >
          <Activity className="w-4 h-4" />
          Make Predictions
        </button>
        <button
          onClick={() => setActiveTab('batch')}
          className={`px-4 py-2 font-semibold transition-all flex items-center gap-2 ${
            activeTab === 'batch'
              ? 'text-purple-400 border-b-2 border-purple-400'
              : 'text-gray-400 hover:text-gray-300'
          }`}
          disabled={!model_id}
        >
          <Upload className="w-4 h-4" />
          Batch Predictions
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'results' && (
        <div className="space-y-6">

      {/* Dataset Info */}
      {dataset_info && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Dataset Size</div>
            <div className="text-2xl font-bold text-white">
              {dataset_info.original_shape?.[0] || 'N/A'}
            </div>
            <div className="text-xs text-gray-500">rows</div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Features</div>
            <div className="text-2xl font-bold text-white">
              {dataset_info.original_shape?.[1] || 'N/A'}
            </div>
            <div className="text-xs text-gray-500">columns</div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Task Type</div>
            <div className="text-lg font-bold text-purple-400 capitalize">
              {dataset_info.task_type || 'Unknown'}
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Target</div>
            <div className="text-lg font-bold text-blue-400">
              {dataset_info.target_column || 'N/A'}
            </div>
          </div>
        </div>
      )}

      {/* Model Performance */}
      <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-lg p-6 border border-purple-500/30">
        <div className="flex items-center gap-2 mb-4">
          <Target className="w-6 h-6 text-purple-400" />
          <h3 className="text-lg font-bold text-white">Best Model Performance</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-400 mb-1">Model</div>
            <div className="text-xl font-bold text-white">{best_model_name || 'N/A'}</div>
          </div>

          <div>
            <div className="text-sm text-gray-400 mb-1">CV Score (Training)</div>
            <div className="text-3xl font-bold text-green-400">
              {best_cv_score ? `${(best_cv_score * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </div>

          {best_test_score && (
            <div>
              <div className="text-sm text-gray-400 mb-1">Test Score</div>
              <div className="text-3xl font-bold text-blue-400">
                {(best_test_score * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>

        {/* Performance Indicator */}
        {best_cv_score && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-gray-400">Accuracy</span>
              <span className="text-white font-semibold">
                {best_cv_score >= 0.85 ? '🟢 Excellent' :
                 best_cv_score >= 0.75 ? '🟡 Good' :
                 best_cv_score >= 0.60 ? '🟠 Fair' : '🔴 Needs Improvement'}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all ${
                  best_cv_score >= 0.85 ? 'bg-green-500' :
                  best_cv_score >= 0.75 ? 'bg-yellow-500' :
                  best_cv_score >= 0.60 ? 'bg-orange-500' : 'bg-red-500'
                }`}
                style={{ width: `${best_cv_score * 100}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Model Selection Reasoning */}
      {model_selection_metadata && (
        <div className="bg-gradient-to-br from-indigo-900/30 to-blue-900/30 rounded-lg p-5 border border-indigo-500/30">
          <div className="flex items-center gap-2 mb-4">
            <Target className="w-6 h-6 text-indigo-400" />
            <h3 className="text-lg font-bold text-white">Model Selection Strategy</h3>
          </div>

          {/* Data Characteristics & Tier */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="text-xs text-gray-400 mb-2">Dataset Characteristics</div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Samples:</span>
                  <span className="text-white font-semibold">
                    {model_selection_metadata.data_characteristics?.n_samples?.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Features:</span>
                  <span className="text-white font-semibold">
                    {model_selection_metadata.data_characteristics?.n_features}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="text-xs text-gray-400 mb-2">Selection Tier</div>
              <div className="text-2xl font-bold text-indigo-400 capitalize mb-1">
                {model_selection_metadata.selection_tier} Dataset
              </div>
              <div className="text-xs text-gray-500">
                {model_selection_metadata.selection_tier === 'small' && '<1,000 samples'}
                {model_selection_metadata.selection_tier === 'medium' && '1K-100K samples'}
                {model_selection_metadata.selection_tier === 'large' && '>100K samples'}
              </div>
            </div>
          </div>

          {/* Reasoning */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
            <div className="text-xs text-blue-300 font-semibold mb-2">🎯 Selection Reasoning</div>
            <div className="text-sm text-white">
              {model_selection_metadata.reasoning}
            </div>
          </div>

          {/* Model Rationale */}
          {model_selection_metadata.model_rationale && Object.keys(model_selection_metadata.model_rationale).length > 0 && (
            <div>
              <div className="text-xs text-gray-400 mb-3">Why These Models Were Chosen:</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(model_selection_metadata.model_rationale).map(([model, reason]) => (
                  <div key={model} className="bg-gray-900/50 rounded-lg p-3">
                    <div className="font-semibold text-white text-sm mb-1">{model}</div>
                    <div className="text-xs text-gray-400">{reason}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* High Cardinality Warning */}
          {model_selection_metadata.data_characteristics?.has_high_cardinality && (
            <div className="mt-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
              <div className="text-xs text-yellow-300">
                ⚠️ <strong>Note:</strong> High-cardinality features detected in{' '}
                {model_selection_metadata.data_characteristics.high_cardinality_columns.length} column(s).
                CatBoost was included for optimal handling.
              </div>
            </div>
          )}
        </div>
      )}

      {/* Enhanced Observability Panel */}
      {observability && (
        <div className="space-y-4">
          {/* Pipeline Performance */}
          <div className="bg-gradient-to-br from-gray-800/80 to-gray-900/80 rounded-lg p-5 border border-gray-700">
            <div className="flex items-center gap-2 mb-4">
              <Zap className="w-6 h-6 text-yellow-400" />
              <h3 className="text-lg font-bold text-white">Pipeline Performance</h3>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Duration */}
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Total Duration</div>
                <div className="text-2xl font-bold text-white">
                  {observability.duration_seconds?.toFixed(1)}s
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {observability.duration_seconds < 90 ? '🟢 Fast' :
                   observability.duration_seconds < 180 ? '🟡 Normal' : '🔴 Slow'}
                </div>
              </div>

              {/* Tokens */}
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Total Tokens</div>
                <div className="text-2xl font-bold text-white">
                  {observability.estimated_tokens?.toLocaleString()}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  ~{(observability.estimated_tokens / observability.duration_seconds).toFixed(0)} tok/s
                </div>
              </div>

              {/* Cost */}
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Pipeline Cost</div>
                <div className="text-2xl font-bold text-green-400">
                  ${observability.estimated_cost_usd?.toFixed(4)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {observability.estimated_cost_usd < 0.05 ? '🟢 Low' :
                   observability.estimated_cost_usd < 0.15 ? '🟡 Medium' : '🟠 High'}
                </div>
              </div>

              {/* Correlation ID */}
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Trace ID</div>
                <div className="text-lg font-mono font-bold text-purple-400">
                  {observability.correlation_id?.substring(0, 8) || 'N/A'}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {observability.correlation_id ? 'End-to-end tracking' : 'Tracking disabled'}
                </div>
              </div>
            </div>
          </div>

          {/* Cache Performance */}
          {observability.cache_stats && (
            <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-lg p-5 border border-blue-500/30">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-6 h-6 text-blue-400" />
                <h3 className="text-lg font-bold text-white">LLM Cache Performance</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Cache Hit Rate */}
                <div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-gray-400">Cache Hit Rate</span>
                    <span className="text-white font-semibold">
                      {observability.cache_stats.hit_rate_percent?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full transition-all"
                      style={{ width: `${observability.cache_stats.hit_rate_percent}%` }}
                    ></div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-gray-400">Hits</div>
                      <div className="text-green-400 font-semibold">
                        {observability.cache_stats.hits}
                      </div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-gray-400">Misses</div>
                      <div className="text-orange-400 font-semibold">
                        {observability.cache_stats.misses}
                      </div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-gray-400">Total</div>
                      <div className="text-white font-semibold">
                        {observability.cache_stats.total_requests}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Cost Savings */}
                <div>
                  <div className="text-sm text-gray-400 mb-2">Cache Cost Savings</div>
                  <div className="text-4xl font-bold text-green-400 mb-2">
                    ${observability.cache_stats.cost_saved_usd?.toFixed(4)}
                  </div>
                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                    <div className="text-xs text-green-300 mb-1">Efficiency Gain</div>
                    <div className="text-sm text-white">
                      {observability.cache_stats.hit_rate_percent > 0 ? (
                        <>
                          Saved {observability.cache_stats.hit_rate_percent.toFixed(0)}% of LLM costs
                          {observability.cache_stats.hits > 0 && (
                            <span className="block text-xs text-gray-400 mt-1">
                              by reusing {observability.cache_stats.hits} cached decision{observability.cache_stats.hits !== 1 ? 's' : ''}
                            </span>
                          )}
                        </>
                      ) : (
                        'First run - building cache'
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Cache Insight */}
              {observability.cache_stats.hit_rate_percent === 0 && observability.cache_stats.total_requests > 0 && (
                <div className="mt-4 bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                  <div className="text-xs text-blue-300">
                    💡 <strong>Tip:</strong> Run the same dataset again to see cache in action.
                    Future runs with similar data will be ~80% cheaper and faster!
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Cost Breakdown */}
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h4 className="text-sm font-bold text-white mb-3">Cost Analysis</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <div className="text-gray-400 mb-1">Actual Cost</div>
                <div className="text-xl font-bold text-white">
                  ${observability.estimated_cost_usd?.toFixed(4)}
                </div>
              </div>
              {observability.cache_stats && (
                <>
                  <div>
                    <div className="text-gray-400 mb-1">Cache Savings</div>
                    <div className="text-xl font-bold text-green-400">
                      -${observability.cache_stats.cost_saved_usd?.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 mb-1">Without Cache</div>
                    <div className="text-xl font-bold text-gray-500">
                      ${(observability.estimated_cost_usd + observability.cache_stats.cost_saved_usd)?.toFixed(4)}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Download Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button
          onClick={handleDownloadReport}
          disabled={!results?.report_filename}
          className="py-3 px-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg text-white font-semibold flex items-center justify-center gap-2 transition-all"
        >
          <Download className="w-5 h-5" />
          Download Report (HTML)
        </button>

        <button
          onClick={handleDownloadModel}
          disabled={!model_id}
          className="py-3 px-4 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg text-white font-semibold flex items-center justify-center gap-2 transition-all"
        >
          <Package className="w-5 h-5" />
          Download Model Package
        </button>
      </div>

        {/* Next Steps */}
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <h3 className="text-sm font-bold text-blue-300 mb-2">🎯 Next Steps:</h3>
          <ul className="text-xs text-blue-200 space-y-1">
            <li>✓ Model is trained and ready for predictions</li>
            <li>✓ Use this model to predict new data</li>
            <li>✓ Deploy to production via API</li>
            <li>✓ Download full report for stakeholders</li>
          </ul>
        </div>
        </div>
      )}

      {/* Single Prediction Tab */}
      {activeTab === 'predict' && model_id && feature_names && (
        <PredictionInterface
          modelId={model_id}
          featureNames={feature_names}
          taskType={dataset_info?.task_type || 'classification'}
          onPredictionComplete={(prediction) => {
            console.log('Prediction complete:', prediction)
          }}
        />
      )}

      {/* Batch Prediction Tab */}
      {activeTab === 'batch' && model_id && (
        <BatchPrediction
          modelId={model_id}
          taskType={dataset_info?.task_type || 'classification'}
          onPredictionComplete={(result) => {
            console.log('Batch prediction complete:', result)
          }}
        />
      )}
    </div>
  )
}
