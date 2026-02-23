Write-Host "=== Knowledge GraphRAG 환경 초기화 ===" -ForegroundColor Cyan
Write-Host ""

# uv 설치 확인
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[!] uv가 설치되어 있지 않습니다. 설치합니다..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
}
Write-Host "[OK] uv $(uv --version)" -ForegroundColor Green

# .env 파일 생성
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[OK] .env 파일 생성 완료 (.env.example 복사)" -ForegroundColor Green
    Write-Host "     -> .env 파일을 열어 NEO4J_PASSWORD를 설정하세요" -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env 파일 이미 존재" -ForegroundColor Green
}

# uv sync (가상환경 + 의존성 설치)
Write-Host ""
Write-Host "패키지 설치 중 (uv sync)..." -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] 가상환경 및 패키지 설치 완료" -ForegroundColor Green
} else {
    Write-Host "[!] uv sync 실패. 에러를 확인하세요." -ForegroundColor Red
    exit 1
}

# Ollama 확인
Write-Host ""
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Ollama 설치됨" -ForegroundColor Green
    $models = ollama list 2>&1
    if ($models -match "qwen3:8b-q4_K_M") {
        Write-Host "[OK] qwen3:8b-q4_K_M 모델 있음" -ForegroundColor Green
    } else {
        Write-Host "[!] LLM 모델 다운로드가 필요합니다:" -ForegroundColor Yellow
        Write-Host "     ollama pull qwen3:8b-q4_K_M" -ForegroundColor White
    }
    if ($models -match "bge-m3-korean") {
        Write-Host "[OK] bona/bge-m3-korean 모델 있음" -ForegroundColor Green
    } else {
        Write-Host "[!] Embedding 모델 다운로드가 필요합니다:" -ForegroundColor Yellow
        Write-Host "     ollama pull bona/bge-m3-korean" -ForegroundColor White
    }
} else {
    Write-Host "[!] Ollama가 설치되어 있지 않습니다." -ForegroundColor Yellow
    Write-Host "     https://ollama.ai 에서 설치하세요" -ForegroundColor White
}

# 완료
Write-Host ""
Write-Host "=== 초기화 완료 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "다음 단계:" -ForegroundColor White
Write-Host "  1. .env 파일에서 NEO4J_PASSWORD 설정" -ForegroundColor Gray
Write-Host "  2. Neo4j Desktop 실행 및 DB 시작" -ForegroundColor Gray
Write-Host "  3. Ollama 모델 다운로드 (위 안내 참고)" -ForegroundColor Gray
Write-Host "  4. notebooks/ 폴더의 노트북을 01 -> 02 -> 03 순서로 실행" -ForegroundColor Gray
Write-Host "  5. 웹 서버: uv run python web/app.py" -ForegroundColor Gray
Write-Host ""
