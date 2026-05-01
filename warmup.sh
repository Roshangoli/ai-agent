#!/bin/bash

# Warmup script to keep Render services awake before demo
# Run this every 10 minutes starting 30 min before your demo

BACKEND_URL="https://ai-analytics-backend.onrender.com"
FRONTEND_URL="https://ai-analytics-frontend.onrender.com"

echo "🔥 Warming up AI Analytics Platform..."
echo "Time: $(date '+%I:%M %p')"
echo "================================"

# Wake up backend
echo "1️⃣ Backend API..."
curl -s "$BACKEND_URL/api/health" | head -n 1

# Wake up frontend
echo ""
echo "2️⃣ Frontend..."
curl -s -o /dev/null -w "Status: %{http_code}\n" "$FRONTEND_URL"

echo ""
echo "✅ Warmup complete! Services are ready."
echo "🕐 Run again in 10 minutes"
echo "================================"
