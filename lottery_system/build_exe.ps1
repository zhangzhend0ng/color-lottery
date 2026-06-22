# lottery_system\build_exe.ps1
# Run this in PowerShell to produce a standalone lottery-server.exe
# Output: server\dist\lottery-server.exe (single file, no Python needed)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot\server

# 1. Ensure PyInstaller is installed
python -m pip install pyinstaller flask flask-cors 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot install dependencies. Check Python/pip." -ForegroundColor Red
    Pop-Location; exit 1
}

# 2. Build
Write-Host "Building lottery-server.exe..." -ForegroundColor Cyan
pyinstaller `
    --onefile `
    --console `
    --name "lottery-server" `
    --add-data "static;static" `
    --add-data "..\color_detector\color_mapping.json;." `
    app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    Pop-Location; exit 1
}

# 3. Result
$exe = "$PSScriptRoot\server\dist\lottery-server.exe"
if (Test-Path $exe) {
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Done: $exe ($size MB)" -ForegroundColor Green
    Write-Host ""
    Write-Host "To use on another PC:" -ForegroundColor Yellow
    Write-Host "  1. Copy lottery-server.exe to the other PC"
    Write-Host "  2. Double-click to start (or run in terminal)"
    Write-Host "  3. Phone opens http://<this-PC-IP>:5000/detect"
    Write-Host "  4. Display opens http://<this-PC-IP>:5000"
}

Pop-Location