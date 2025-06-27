#!/bin/bash

# Exit if no commit message is provided
if [ -z "$1" ]; then
  echo "❌ Please provide a commit message!"
  echo "Usage: ./deploy.sh 'Your commit message'"
  exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Stage, commit, push
git add .
git commit -m "$1"
git push

# Deploy site
mkdocs gh-deploy --clean
