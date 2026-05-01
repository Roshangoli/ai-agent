import {
  DollarSign,
  ShoppingCart,
  MapPin,
  Package,
  Database,
  Columns,
  Hash,
  Type,
  Layers,
  Calendar,
  CheckCircle,
  Activity,
  Info
} from 'lucide-react'

const StatsPanel = ({ stats, isDynamic }) => {
  if (!stats) return null

  // Icon mapping for dynamic stats
  const iconMap = {
    'database': Database,
    'columns': Columns,
    'hash': Hash,
    'text': Type,
    'dollar-sign': DollarSign,
    'layers': Layers,
    'calendar': Calendar,
    'check-circle': CheckCircle,
    'activity': Activity,
    'info': Info,
    'shopping-cart': ShoppingCart,
    'map-pin': MapPin,
    'package': Package
  }

  const colorMap = ['text-green-400', 'text-blue-400', 'text-purple-400', 'text-orange-400']
  const bgColorMap = ['bg-green-500/10', 'bg-blue-500/10', 'bg-purple-500/10', 'bg-orange-500/10']

  let statCards = []

  // Handle dynamic stats (new format)
  if (isDynamic && typeof stats === 'object' && !stats.stats) {
    statCards = Object.entries(stats).map(([key, stat], index) => ({
      label: stat.label,
      value: stat.value,
      icon: iconMap[stat.icon] || Database,
      color: colorMap[index % colorMap.length],
      bgColor: bgColorMap[index % bgColorMap.length]
    }))
  }
  // Handle old stats format (from /api/stats)
  else if (stats.stats) {
    const data = stats.stats
    statCards = [
      {
        label: 'Total Sales',
        value: data.total_sales
          ? `$${Math.round(data.total_sales.total).toLocaleString()}`
          : '-',
        icon: DollarSign,
        color: 'text-green-400',
        bgColor: 'bg-green-500/10',
      },
      {
        label: 'Transactions',
        value: data.total_transactions
          ? data.total_transactions.count.toLocaleString()
          : '-',
        icon: ShoppingCart,
        color: 'text-blue-400',
        bgColor: 'bg-blue-500/10',
      },
      {
        label: 'Regions',
        value: data.regions ? data.regions.count : '-',
        icon: MapPin,
        color: 'text-purple-400',
        bgColor: 'bg-purple-500/10',
      },
      {
        label: 'Products',
        value: data.products ? data.products.count : '-',
        icon: Package,
        color: 'text-orange-400',
        bgColor: 'bg-orange-500/10',
      },
    ]
  }

  return (
    <div>
      {/* Dynamic stats indicator */}
      {isDynamic && (
        <div className="mb-3 flex items-center justify-center gap-2 text-sm">
          <span className="px-3 py-1 bg-green-500/20 text-green-300 rounded-full border border-green-500/30">
            ✨ Stats Auto-Generated from Your CSV
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat, index) => {
          const Icon = stat.icon
          return (
            <div
              key={index}
              className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 shadow-lg"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm font-medium">
                    {stat.label}
                  </p>
                  <p className="text-white text-2xl font-bold mt-1">
                    {stat.value}
                  </p>
                </div>
                <div className={`${stat.bgColor} p-3 rounded-lg`}>
                  <Icon className={`w-6 h-6 ${stat.color}`} />
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default StatsPanel