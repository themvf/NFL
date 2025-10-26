# NFL Data Viewer - Streamlit App Launcher
# This script runs the pfr_viewer.py Streamlit application

# Change to the script's directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Working Directory: $scriptPath" -ForegroundColor Cyan
Write-Host "Starting NFL Data Viewer (Streamlit)..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Check if streamlit is available via Python module
$streamlitInstalled = $false
try {
    $result = python -m streamlit --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $streamlitInstalled = $true
        Write-Host "Streamlit found: $result" -ForegroundColor Green
        Write-Host ""
    }
} catch {
    $streamlitInstalled = $false
}

if (-not $streamlitInstalled) {
    Write-Host "Streamlit not found. Installing requirements..." -ForegroundColor Yellow
    Write-Host ""

    # Install requirements
    pip install -r requirements.txt

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host ""
    Write-Host "Requirements installed successfully!" -ForegroundColor Green
    Write-Host ""
}

# Run the Streamlit app using Python module
Write-Host "Launching pfr_viewer.py..." -ForegroundColor Green
Write-Host ""
python -m streamlit run pfr_viewer.py

# If streamlit exits with an error, pause to show the message
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Streamlit exited with an error." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
