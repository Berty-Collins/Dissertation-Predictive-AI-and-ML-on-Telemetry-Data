@echo off
echo ============================================================
echo  DISSERTATION DATA COLLECTION -- BOTH DATASETS
echo  ABS-on (rev-build) then ABS-off (rev-build)
echo  Resumes from existing runs -- safe to re-run if interrupted
echo ============================================================

call .venv\Scripts\activate.bat

echo.
echo [1/2] Starting ABS-ON dataset (runs_rb)...
echo       Target: 555 runs  (skipping already completed)
echo.
python data_collection\scenario_runner.py --rev-build
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ABS-on run failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [1/2] ABS-ON dataset complete.
echo.
echo [2/2] Starting ABS-OFF dataset (no_abs_rb)...
echo       Target: 555 runs  (skipping already completed)
echo.
python data_collection\scenario_runner.py --rev-build --abs-off
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ABS-off run failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo  Both datasets complete.
echo  Results:
echo    ABS-on  -> results\sweep_results_rb.csv
echo    ABS-off -> results\sweep_results_no_abs_rb.csv
echo ============================================================
pause
