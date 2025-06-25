@echo off
REM === Check if a commit message is provided ===
IF "%~1"=="" (
    echo ❌ Please provide a commit message!
    echo Usage: deploy.bat "Your commit message"
    exit /b 1
)

REM === Stage, commit, and push changes ===
git add .
git commit -m "%~1"
git push

REM === Deploy site using MkDocs ===
mkdocs gh-deploy --clean
