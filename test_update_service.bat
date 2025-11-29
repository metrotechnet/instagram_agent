@echo off
REM Test script for Instagram Agent Update Service
echo Starting Instagram Agent Update Service Tests...

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at .venv\Scripts\activate.bat
    echo Using system Python...
)

REM Run the test script
echo.
echo Running tests...
python test_update_service.py %*

REM Pause to see results
echo.
pause