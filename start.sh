#!/bin/bash
# Polymarket BTC Auto-Trader Launcher

echo "🎯 Polymarket BTC 15m Auto-Trader"
echo "================================="
echo ""
echo "Options:"
echo "  1) Auto-Trader (Python) - Trades automatically"
echo "  2) Display Only (Node.js) - Just shows predictions"
echo ""

read -p "Enter choice [1/2]: " choice

cd /home/yogesh/clawd

case $choice in
    1)
        echo "Starting BTC Auto-Trader..."
        source venv/bin/activate
        python btc_auto_trader.py
        ;;
    2)
        echo "Starting BTC Display Assistant..."
        cd btc_assistant
        npm start
        ;;
    *)
        echo "Invalid choice. Running auto-trader..."
        source venv/bin/activate
        python btc_auto_trader.py
        ;;
esac
