# Polymarket Trading Suite 🤖

Your automated Polymarket trading setup with **two powerful tools**:

## Tools

### 1. Arbitrage Bot (`arbitrage_bot.py`)
**Guaranteed profit** trading - finds markets where YES + NO < $1

| Feature | Description |
|---------|-------------|
| Strategy | Buy both YES and NO when combined cost < $1 |
| Risk | **Zero** (if both orders fill) |
| Profit | Difference between cost and $1 payout |
| Markets | Any Polymarket market |

```bash
# Run
source venv/bin/activate
python arbitrage_bot.py
```

### 2. BTC 15m Assistant (`btc_assistant/`)
**Prediction-based** trading for BTC UP/DOWN 15-minute markets

| Feature | Description |
|---------|-------------|
| Strategy | TA prediction (RSI, MACD, VWAP, Heiken Ashi) |
| Risk | **Medium** (predictions can be wrong) |
| Edge | Only trades when model prob > market price |
| Markets | BTC 15-minute UP/DOWN only |

```bash
# Run
cd btc_assistant && npm start
```

---

## Quick Start

### Option 1: Easy Launcher
```bash
chmod +x start.sh
./start.sh
```

### Option 2: Run Manually

**Arbitrage Bot:**
```bash
source venv/bin/activate
python arbitrage_bot.py
```

**BTC Assistant:**
```bash
cd btc_assistant && npm start
```

---

## Configuration

### Arbitrage Bot (`.env`)
```
PRIVATE_KEY=0x...           # Your wallet key
MIN_PROFIT_MARGIN=0.03      # 3% minimum profit
MAX_BET_SIZE=1.50           # $1.50 per side
PAPER_TRADING=true          # Start with paper trading!
```

### BTC Assistant
The assistant is read-only (displays predictions). It doesn't execute trades automatically.

---

## Safety Recommendations

1. **Start with paper trading** (`PAPER_TRADING=true`)
2. **Small amounts first** - You're starting with $3, which is perfect
3. **Monitor both tools** before trusting fully
4. **Understand the difference**:
   - Arbitrage = guaranteed profit, rare opportunities
   - BTC Assistant = predictions, more frequent but has risk

---

## Files

| File | Description |
|------|-------------|
| `arbitrage_bot.py` | Main arbitrage trading bot |
| `btc_assistant/` | BTC 15m prediction assistant |
| `.env` | Your private config (don't share!) |
| `start.sh` | Easy launcher script |
| `venv/` | Python virtual environment |
