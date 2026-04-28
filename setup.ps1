# One-command local setup for Windows
# Run: .\setup.ps1

Write-Host "==> Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

Write-Host "==> Upgrading pip..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "==> Installing dependencies..." -ForegroundColor Cyan
.\venv\Scripts\pip install -r requirements.txt

Write-Host ""
Write-Host "==> Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate venv:      .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run pipeline:       python scripts/run_pipeline.py --stage all"
Write-Host "  3. Start all services: docker-compose up"
Write-Host ""
Write-Host "Service URLs (after docker-compose up):" -ForegroundColor Yellow
Write-Host "  API:        http://localhost:8000"
Write-Host "  API Docs:   http://localhost:8000/docs"
Write-Host "  MLflow:     http://localhost:5000"
Write-Host "  Grafana:    http://localhost:3000  (admin / mlops123)"
Write-Host "  Prometheus: http://localhost:9090"
