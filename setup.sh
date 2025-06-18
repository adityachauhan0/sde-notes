#!/bin/bash

# Exit on any error
set -e

echo "📦 Installing system dependencies..."
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm python python-pip git

echo "🐍 Creating virtual environment..."
python -m venv venv

echo "✅ Activating virtual environment..."
source venv/bin/activate

echo "⬇️ Installing Python packages..."
pip install --upgrade pip
pip install mkdocs mkdocs-material

echo "📝 Saving requirements.txt..."
pip freeze > requirements.txt

echo "✅ Setup complete! Run with: ./deploy.sh 'Your message'"
