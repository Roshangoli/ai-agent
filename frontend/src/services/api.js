import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for AI processing
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.config.url}`, response.data)
    return response
  },
  (error) => {
    console.error(`❌ API Error: ${error.config?.url}`, error.response?.data || error.message)
    return Promise.reject(error)
  }
)

/**
 * Submit a natural language query to the AI analytics system
 */
export const submitQuery = async (question, useLangChain = true, dbPath = null) => {
  try {
    const response = await api.post('/api/query', {
      question,
      use_langchain: useLangChain,
      db_path: dbPath,
    })
    return response.data
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Query failed')
  }
}

/**
 * Upload CSV file for Query Mode
 * Converts CSV to SQLite and returns database path
 */
export const uploadCSVForQuery = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/api/upload/csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 1 minute for CSV conversion
    })

    return response.data
  } catch (error) {
    const errorMessage = error.response?.data?.detail || 'CSV upload failed'
    throw new Error(errorMessage)
  }
}

/**
 * Execute a SQL query directly
 */
export const executeSQLQuery = async (sql) => {
  try {
    const response = await api.post('/api/sql/execute', { sql })
    return response.data
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'SQL execution failed')
  }
}

/**
 * Get database schema
 */
export const getSchema = async () => {
  try {
    const response = await api.get('/api/schema')
    return response.data
  } catch (error) {
    throw new Error('Failed to fetch schema')
  }
}

/**
 * Get database statistics
 */
export const getStats = async () => {
  try {
    const response = await api.get('/api/stats')
    return response.data
  } catch (error) {
    throw new Error('Failed to fetch stats')
  }
}

/**
 * Check API health
 */
export const getHealth = async () => {
  try {
    const response = await api.get('/api/health')
    return response.data
  } catch (error) {
    throw new Error('Health check failed')
  }
}

/**
 * Generate chart from data
 */
export const generateChart = async (data, chartType = 'auto', title = null) => {
  try {
    const response = await api.post('/api/chart/generate', {
      data,
      chart_type: chartType,
      title,
    })
    return response.data
  } catch (error) {
    throw new Error('Chart generation failed')
  }
}

/**
 * Initialize database with sample data
 */
export const initializeDatabase = async () => {
  try {
    const response = await api.post('/api/init/database')
    return response.data
  } catch (error) {
    throw new Error('Database initialization failed')
  }
}

/**
 * Run Data Science Mode pipeline
 * Upload CSV and train ML model autonomously
 */
export const runDataScience = async (file, targetColumn, taskType = 'auto') => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('target_column', targetColumn)
    formData.append('task_type', taskType)

    const response = await api.post('/api/datascience/train', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes for ML training
    })

    return response.data
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.response?.data?.error || 'Data Science pipeline failed'
    throw new Error(errorMessage)
  }
}

export default api