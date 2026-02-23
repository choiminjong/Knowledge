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

# 완료
Write-Host ""
Write-Host "=== 초기화 완료 ===" -ForegroundColor Cyan
Write-Host ""
