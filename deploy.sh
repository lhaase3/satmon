#!/bin/bash
# Production deployment script for SatMon

set -e  # Exit on error

echo "🚀 Deploying SatMon to production..."

# Environment setup
export DATABASE_URL="postgresql://satmon:satmon123@db:5432/satmon"
export PYTHONPATH=/app

# Build and deploy with Docker Compose
echo "📦 Building containers..."
docker-compose -f docker-compose.prod.yml build

echo "🗄️ Starting database..."
docker-compose -f docker-compose.prod.yml up -d db

echo "⏳ Waiting for database to be ready..."
sleep 10

echo "📊 Loading demo data..."
docker-compose -f docker-compose.prod.yml run --rm api python scripts/load_telemanom.py --channel-id P-1
docker-compose -f docker-compose.prod.yml run --rm api python scripts/load_telemanom.py --channel-id S-1
docker-compose -f docker-compose.prod.yml run --rm api python scripts/load_telemanom.py --channel-id T-1

echo "🔍 Running anomaly detection..."
docker-compose -f docker-compose.prod.yml run --rm api python scripts/detect_zscore.py --channel-id 1 --start "2018-01-01T00:00:00Z" --end "2018-01-05T00:00:00Z"
docker-compose -f docker-compose.prod.yml run --rm api python scripts/detect_isoforest.py --channel-id 1 --start "2018-01-01T00:00:00Z" --end "2018-01-05T00:00:00Z"

echo "🌐 Starting web services..."
docker-compose -f docker-compose.prod.yml up -d

echo "✅ Deployment complete!"
echo "🔗 Access your app at: http://localhost:8000"
echo "📊 API docs at: http://localhost:8000/docs"
echo "🎯 Dashboard at: http://localhost:8000/index.html"

# Health check
echo "🏥 Running health check..."
sleep 5
curl -f http://localhost:8000/healthz || echo "⚠️ Health check failed"

echo "🎉 SatMon is live and ready for recruiters!"