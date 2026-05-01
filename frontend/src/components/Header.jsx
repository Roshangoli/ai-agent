import { BarChart3, Activity } from 'lucide-react'

const Header = ({ health }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500'
      case 'degraded':
        return 'bg-yellow-500'
      case 'unhealthy':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <header className="bg-gray-900/50 backdrop-blur-sm border-b border-gray-800">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">
                AI Analytics Dashboard
              </h1>
              <p className="text-sm text-gray-400">
                Multi-Agent Data Analytics System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${getStatusColor(
                health.status
              )} animate-pulse`}
            ></div>
            <span className="text-sm text-gray-300 capitalize">
              {health.status}
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header