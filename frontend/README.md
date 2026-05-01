# React Frontend for AI Analytics System

Modern React frontend for the Multi-Agent AI Data Analytics System.

## Features

- 🎨 Modern UI with Tailwind CSS
- ⚡ Fast development with Vite
- 🤖 Real-time AI query processing
- 📊 Interactive data visualizations
- 🔄 RESTful API integration
- 📱 Responsive design

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - API client
- **Lucide Icons** - Icon library
- **Recharts** - Data visualization (optional)

## Quick Start

### Prerequisites

- Node.js >= 16.0.0
- npm >= 8.0.0
- Backend API running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

Make sure the FastAPI backend is running:

```bash
# From project root
cd backend/api
pip install fastapi uvicorn python-dotenv
python main.py
```

Backend will run on `http://localhost:8000`

## Development

```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── Header.jsx
│   │   ├── QueryPanel.jsx
│   │   ├── ResultsPanel.jsx
│   │   └── StatsPanel.jsx
│   ├── services/         # API services
│   │   └── api.js
│   ├── App.jsx           # Main app component
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles
├── public/               # Static assets
├── index.html            # HTML template
├── package.json          # Dependencies
├── vite.config.js        # Vite configuration
├── tailwind.config.js    # Tailwind configuration
└── README.md             # This file
```

## API Integration

The frontend communicates with the FastAPI backend via REST API:

- **POST /api/query** - Submit natural language questions
- **GET /api/health** - Check API health
- **GET /api/stats** - Get database statistics
- **GET /api/schema** - Get database schema
- **POST /api/sql/execute** - Execute SQL directly

See `src/services/api.js` for full API client implementation.

## Components

### Header
- Displays app title and health status
- Real-time connection indicator

### QueryPanel
- Natural language input
- Example questions
- Submit button with loading state

### ResultsPanel
- AI-generated insights
- Data visualizations
- SQL query display
- Data table preview

### StatsPanel
- Real-time statistics cards
- Total sales, transactions, regions, products

## Environment Variables

```env
# API base URL
VITE_API_URL=http://localhost:8000

# Environment
VITE_ENV=development
```

## Building for Production

```bash
# Build optimized production bundle
npm run build

# Output will be in dist/ directory
# Deploy dist/ folder to your hosting service
```

## Deployment

### Option 1: AWS Amplify

```bash
# See amplify.yml in project root
# Connect GitHub repository to AWS Amplify
# Auto-deploys on push to main branch
```

### Option 2: Vercel

```bash
npm install -g vercel
vercel deploy
```

### Option 3: Netlify

```bash
npm install -g netlify-cli
netlify deploy --prod
```

## Features Roadmap

- [x] Natural language query interface
- [x] Real-time AI processing
- [x] Statistics dashboard
- [x] Responsive design
- [ ] Advanced chart visualizations (Recharts)
- [ ] Query history
- [ ] Export to CSV/Excel
- [ ] User authentication
- [ ] Dark/Light mode toggle
- [ ] Saved queries
- [ ] Real-time updates via WebSocket

## Troubleshooting

### API Connection Issues

If frontend can't connect to backend:
1. Check backend is running: `http://localhost:8000`
2. Verify CORS is enabled in backend
3. Check `.env` file has correct `VITE_API_URL`

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules
npm install

# Clear Vite cache
rm -rf node_modules/.vite
npm run dev
```

### Port Already in Use

```bash
# Change port in vite.config.js
server: {
  port: 3001  // Change from 3000
}
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

MIT License - See main project LICENSE file

---

**Built with ❤️ using React + Vite + Tailwind CSS**