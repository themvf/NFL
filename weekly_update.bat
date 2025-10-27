@echo off
REM Weekly Database Update Workflow
REM This script manages the database sync before/after scraping

echo ==========================================
echo NFL Data - Weekly Update Workflow
echo ==========================================
echo.

:MENU
echo What do you want to do?
echo.
echo 1. Download database from GCS (do this FIRST, before scraping)
echo 2. Upload database to GCS (do this AFTER scraping)
echo 3. Exit
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" goto DOWNLOAD
if "%choice%"=="2" goto UPLOAD
if "%choice%"=="3" goto END
echo Invalid choice. Please try again.
echo.
goto MENU

:DOWNLOAD
echo.
echo Downloading database from GCS...
echo This preserves any injuries/notes added through the app.
echo.
python sync_database.py download
if errorlevel 1 (
    echo.
    echo Download failed! Check the error above.
    pause
    goto MENU
)
echo.
echo ==========================================
echo NEXT STEP: Run your scraping scripts now
echo Location: C:\Docs\_AI Python Projects\Cursor Projects\Scrape and Excel
echo ==========================================
echo.
pause
goto MENU

:UPLOAD
echo.
echo Uploading database to GCS...
echo.
python sync_database.py upload
if errorlevel 1 (
    echo.
    echo Upload failed! Check the error above.
    pause
    goto MENU
)
echo.
echo ==========================================
echo SUCCESS! Database uploaded to GCS
echo Streamlit Cloud will use the updated database on next restart.
echo ==========================================
echo.
pause
goto MENU

:END
echo.
echo Goodbye!
