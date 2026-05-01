import { useState } from 'react'
import { Target, TrendingUp, AlertCircle, BarChart3, Loader2 } from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function PredictionInterface({ modelId, featureNames, taskType, onPredictionComplete }) {
  const [formData, setFormData] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Handle input change
  const handleInputChange = (featureName, value) => {
    setFormData(prev => ({
      ...prev,
      [featureName]: value
    }))
  }

  // Reset form
  const handleReset = () => {
    setFormData({})
    setPrediction(null)
    setError(null)
  }

  // Make prediction
  const handlePredict = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      // Validate all fields are filled
      const missingFields = featureNames.filter(name => !formData[name] && formData[name] !== 0)
      if (missingFields.length > 0) {
        setError(`Please fill in all required fields: ${missingFields.join(', ')}`)
        setLoading(false)
        return
      }

      // Convert numeric fields to numbers
      const processedData = {}
      for (const [key, value] of Object.entries(formData)) {
        // Try to convert to number if it looks like a number
        const numValue = Number(value)
        processedData[key] = isNaN(numValue) ? value : numValue
      }

      const response = await fetch(`${API_BASE_URL}/api/datascience/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
          data: processedData
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Prediction failed')
      }

      const result = await response.json()
      setPrediction(result)

      if (onPredictionComplete) {
        onPredictionComplete(result)
      }
    } catch (err) {
      setError(err.message || 'Failed to make prediction')
    } finally {
      setLoading(false)
    }
  }

  // Get risk level styling
  const getRiskLevelStyle = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-300'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'low':
        return 'bg-green-100 text-green-800 border-green-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return '⚠️'
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
        <Target className="w-6 h-6 text-purple-400" />
        <div>
          <h3 className="text-xl font-bold text-white">Test Your Model</h3>
          <p className="text-sm text-gray-400">Enter feature values to make a prediction</p>
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

      {/* Form */}
      <form onSubmit={handlePredict} className="space-y-4">
        {/* Feature Inputs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {featureNames && featureNames.map((featureName) => (
            <div key={featureName}>
              <label className="block text-sm font-medium text-gray-300 mb-2 capitalize">
                {featureName.replace(/_/g, ' ')}
              </label>
              <input
                type="text"
                value={formData[featureName] || ''}
                onChange={(e) => handleInputChange(featureName, e.target.value)}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder={`Enter ${featureName.replace(/_/g, ' ')}`}
                required
              />
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 py-3 px-6 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg text-white font-semibold flex items-center justify-center gap-2 transition-all"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <Target className="w-5 h-5" />
                Predict
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="py-3 px-6 bg-gray-700 hover:bg-gray-600 rounded-lg text-white font-semibold transition-all"
          >
            Reset
          </button>
        </div>
      </form>

      {/* Prediction Results */}
      {prediction && (
        <div className="mt-6 pt-6 border-t border-gray-700 space-y-4">
          <h4 className="text-lg font-bold text-white mb-4">📊 Prediction Result</h4>

          {/* Main Prediction */}
          {taskType === 'classification' && prediction.risk_level && (
            <div className={`p-6 rounded-lg border-2 ${getRiskLevelStyle(prediction.risk_level)}`}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-semibold opacity-75 mb-1">Risk Level</div>
                  <div className="text-3xl font-bold uppercase flex items-center gap-2">
                    <span>{getRiskIcon(prediction.risk_level)}</span>
                    <span>{prediction.risk_level} RISK</span>
                  </div>
                </div>
                {prediction.probability !== null && (
                  <div className="text-right">
                    <div className="text-sm font-semibold opacity-75 mb-1">Confidence</div>
                    <div className="text-3xl font-bold">
                      {(prediction.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Regression Prediction */}
          {taskType === 'regression' && (
            <div className="p-6 bg-blue-900/30 border-2 border-blue-500/50 rounded-lg">
              <div className="text-sm font-semibold text-blue-300 mb-1">Predicted Value</div>
              <div className="text-3xl font-bold text-blue-200">
                {typeof prediction.prediction === 'number'
                  ? prediction.prediction.toFixed(2)
                  : prediction.prediction}
              </div>
            </div>
          )}

          {/* Top Contributing Factors */}
          {prediction.top_factors && prediction.top_factors.length > 0 && (
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h5 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Top Contributing Factors
              </h5>
              <div className="space-y-3">
                {prediction.top_factors.map((factor, index) => (
                  <div key={index}>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-gray-300 capitalize">
                        {factor.feature.replace(/_/g, ' ')}
                      </span>
                      <span className="text-white font-semibold">
                        {(factor.contribution * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                        style={{ width: `${factor.contribution * 100}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      Value: {typeof factor.value === 'number'
                        ? factor.value.toFixed(2)
                        : factor.value}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model Info */}
          <div className="text-xs text-gray-500 flex items-center gap-4">
            <span>Model: {prediction.model_type}</span>
            <span>•</span>
            <span>Task: {prediction.task_type}</span>
          </div>
        </div>
      )}
    </div>
  )
}