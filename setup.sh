#!/bin/bash
# ===========================================
# POLYMARKET ARBITRAGE BOT SETUP
# ===========================================

set -e

echo "🤖 Polymarket Arbitrage Bot Setup"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Python: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "========================================"
echo "NEXT STEPS:"
echo "========================================"
echo ""
echo "1. Edit .env and add your private key:"
echo "   nano .env"
echo ""
echo "2. Start the bot in PAPER TRADING mode:"
echo "   source venv/bin/activate"
echo "   python arbitrage_bot.py"
echo ""
echo "3. Once satisfied, set PAPER_TRADING=false in .env for real trades"
echo ""
echo "========================================"
