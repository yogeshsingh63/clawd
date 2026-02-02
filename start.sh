#!/bin/bash
# Polymarket BTC Trading Bots Launcher

echo "🎯 Polymarket BTC Trading Bots"
echo "=============================="
echo ""
echo "Options:"
echo "  1) TA Bot   - Trades UP or DOWN based on technical analysis"
echo "  2) ARB Bot  - Buys BOTH sides when UP+DOWN < \$1 (guaranteed profit)"
echo "  3) Display  - Just shows predictions (Node.js)"
echo ""
read -p "Enter choice [1/2/3]: " choice

cd "$(dirname "$0")"

case $choice in
    1)
        echo "Starting TA Auto-Trader..."
        source venv/bin/activate 2>/dev/null || true
        python btc_auto_trader.py
        ;;
    2)
        echo "Starting Arbitrage Bot..."
        source venv/bin/activate 2>/dev/null || true
        python btc_arb_bot.py
        ;;
    3)
        echo "Starting Display Only..."
        cd btc_assistant && npm start
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
