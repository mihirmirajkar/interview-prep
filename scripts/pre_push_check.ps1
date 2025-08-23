#!/usr/bin/env pwsh

# Pre-push code quality check script
Write-Host "🔧 Running code quality checks..." -ForegroundColor Cyan

Write-Host "`n1. Formatting code with Black..." -ForegroundColor Yellow
black .

Write-Host "`n2. Sorting imports with isort..." -ForegroundColor Yellow
isort .

Write-Host "`n3. Linting with Ruff..." -ForegroundColor Yellow
ruff check . --fix

Write-Host "`n4. Type checking with MyPy..." -ForegroundColor Yellow
mypy src

Write-Host "`n5. Running tests with coverage..." -ForegroundColor Yellow
pytest --cov=src --cov-report=term-missing

Write-Host "`n✅ All checks complete!" -ForegroundColor Green
Write-Host "Your code is ready to push! 🚀" -ForegroundColor Green
