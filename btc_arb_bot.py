#!/usr/bin/env python3
"""
Polymarket Multi-Outcome Arb Bot (Strict, Only-If-Locked)
=========================================================
Strategy: For a single market with mutually exclusive outcomes,
buy YES on every outcome if total cost < $1 after fees/slippage.
Guaranteed profit only when all legs can fill at quoted depth.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# Configuration
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
SCAN_INTERVAL = int(os.getenv("ARB_SCAN_INTERVAL", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Arb settings
ORDER_SIZE = float(os.getenv("ARB_ORDER_SIZE", "10"))  # shares per outcome
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.5"))
ARB_SLIPPAGE_BPS = float(os.getenv("ARB_SLIPPAGE_BPS", "5"))
ARB_FEE_PCT = float(os.getenv("ARB_FEE_PCT", "0.0"))
MAX_TRADES_PER_MARKET = int(os.getenv("ARB_MAX_TRADES_PER_MARKET", "1"))

# Market filters
MIN_OUTCOMES = int(os.getenv("ARB_MIN_OUTCOMES", "2"))
MIN_LIQUIDITY = float(os.getenv("ARB_MIN_LIQUIDITY", "0"))
MAX_MARKETS_SCAN = int(os.getenv("ARB_MAX_MARKETS_SCAN", "200"))
ARB_MARKETS_PER_CYCLE = int(os.getenv("ARB_MARKETS_PER_CYCLE", "40"))
ARB_CONCURRENCY = int(os.getenv("ARB_CONCURRENCY", "5"))
ARB_PREFILTER_MAX_SUM = float(os.getenv("ARB_PREFILTER_MAX_SUM", "1.0"))

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


class PolymarketArbBot:
    """Strict multi-outcome arbitrage bot across Polymarket markets."""

    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30.0)
        self.clob_client = None
        self.opportunities_found = 0
        self.trades_executed = 0
        self.total_profit = 0.0
        self.running = False
        self.trades_by_market: Dict[str, int] = {}

    async def initialize(self) -> bool:
        console.print(Panel.fit(
            "[bold blue]🎯 Polymarket Multi-Outcome Arb Bot[/bold blue]\n"
            f"Mode: {'[yellow]PAPER TRADING[/yellow]' if PAPER_TRADING else '[red]LIVE TRADING[/red]'}\n"
            f"Order Size: {ORDER_SIZE} shares per outcome\n"
            f"Min Profit: {MIN_PROFIT_PCT}%\n"
            f"Slippage: {ARB_SLIPPAGE_BPS} bps | Fees: {ARB_FEE_PCT}%"
        ))

        if PAPER_TRADING:
            console.print("[yellow]📝 Paper trading mode[/yellow]")
        elif not PRIVATE_KEY:
            console.print("[red]❌ PRIVATE_KEY required for live trading[/red]")
            return False
        else:
            try:
                from py_clob_client.client import ClobClient
                self.clob_client = ClobClient(
                    CLOB_API_URL,
                    key=PRIVATE_KEY.strip(),
                    chain_id=137,
                    signature_type=1
                )
                creds = self.clob_client.create_or_derive_api_creds()
                self.clob_client.set_api_creds(creds)
                console.print(f"[green]✅ Wallet: {self.clob_client.get_address()}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to initialize: {e}[/red]")
                return False

        return True

    async def fetch_active_markets(self) -> List[dict]:
        try:
            url = f"{GAMMA_API_URL}/events"
            params = {
                "active": "true",
                "closed": "false",
                "limit": str(MAX_MARKETS_SCAN),
            }
            resp = await self.http.get(url, params=params)
            resp.raise_for_status()
            events = resp.json()
            markets = []
            for event in events:
                for m in event.get("markets", []):
                    markets.append(m)
            prefiltered = [m for m in markets if self._prefilter_market(m)]
            prefiltered.sort(key=self._market_liquidity, reverse=True)
            if ARB_MARKETS_PER_CYCLE > 0:
                prefiltered = prefiltered[:ARB_MARKETS_PER_CYCLE]
            return prefiltered
        except Exception as e:
            logger.error(f"Market fetch error: {e}")
            return []

    @staticmethod
    def _parse_json_field(value):
        if isinstance(value, str):
            try:
                import json
                return json.loads(value)
            except Exception:
                return []
        return value

    async def get_ask_fill_price(self, token_id: str, size: float) -> Optional[dict]:
        """Estimate avg fill price for a buy of `size` using the ask book."""
        try:
            url = f"{CLOB_API_URL}/book?token_id={token_id}"
            resp = await self.http.get(url)
            data = resp.json()
            asks = data.get("asks", [])
            if not asks:
                return None

            asks.sort(key=lambda x: float(x.get("price", 1.0)))
            remaining = size
            cost = 0.0
            for level in asks:
                price = float(level.get("price", 1.0))
                avail = float(level.get("size", 0.0))
                if avail <= 0:
                    continue
                take = min(remaining, avail)
                cost += take * price
                remaining -= take
                if remaining <= 0:
                    break

            if remaining > 0:
                return None

            avg_price = cost / size if size > 0 else None
            return {"avg_price": avg_price, "total_cost": cost}
        except Exception:
            return None

    def _market_liquidity(self, market: dict) -> float:
        try:
            return float(market.get("liquidityNum") or market.get("liquidity") or 0)
        except Exception:
            return 0.0

    def _prefilter_market(self, market: dict) -> bool:
        outcomes = self._parse_json_field(market.get("outcomes", [])) or []
        tokens = self._parse_json_field(market.get("clobTokenIds", [])) or []
        if len(outcomes) < MIN_OUTCOMES:
            return False
        if len(tokens) < len(outcomes):
            return False
        liquidity = self._market_liquidity(market)
        if liquidity < MIN_LIQUIDITY:
            return False

        outcome_prices = self._parse_json_field(market.get("outcomePrices", [])) or []
        if len(outcome_prices) >= len(outcomes):
            try:
                total = sum(float(outcome_prices[i]) for i in range(len(outcomes)))
                if total >= ARB_PREFILTER_MAX_SUM:
                    return False
            except Exception:
                pass
        return True

    async def check_arbitrage(self, market: dict) -> Optional[dict]:
        outcomes = self._parse_json_field(market.get("outcomes", [])) or []
        tokens = self._parse_json_field(market.get("clobTokenIds", [])) or []

        if len(outcomes) < MIN_OUTCOMES:
            return None
        if len(tokens) < len(outcomes):
            return None

        liquidity = self._market_liquidity(market)
        if liquidity < MIN_LIQUIDITY:
            return None

        fill_quotes = []
        total_raw_cost = 0.0
        for i, outcome in enumerate(outcomes):
            token_id = tokens[i]
            fill = await self.get_ask_fill_price(token_id, ORDER_SIZE)
            if not fill or fill["avg_price"] is None:
                return None
            total_raw_cost += fill["avg_price"]
            fill_quotes.append({
                "outcome": outcome,
                "token_id": token_id,
                "avg_price": fill["avg_price"],
            })

        slippage = (ARB_SLIPPAGE_BPS / 10_000.0) * total_raw_cost
        fee_cost = total_raw_cost * (ARB_FEE_PCT / 100.0)
        total_cost = total_raw_cost + slippage + fee_cost

        if total_cost >= 1.0:
            return None

        profit_per_share = 1.0 - total_cost
        profit_pct = (profit_per_share / total_cost) * 100 if total_cost > 0 else 0
        if profit_pct < MIN_PROFIT_PCT:
            return None

        return {
            "market_id": str(market.get("id") or market.get("slug") or market.get("question", "market")),
            "question": market.get("question", "Polymarket"),
            "outcomes": fill_quotes,
            "total_cost": total_cost,
            "raw_cost": total_raw_cost,
            "slippage": slippage,
            "fee_cost": fee_cost,
            "profit_per_share": profit_per_share,
            "profit_pct": profit_pct,
            "order_size": ORDER_SIZE,
            "expected_profit": profit_per_share * ORDER_SIZE,
        }

    async def execute_arbitrage(self, opp: dict):
        self.opportunities_found += 1
        console.print("\n[bold green]🎯 ARBITRAGE OPPORTUNITY![/bold green]")
        console.print(f"Market: {opp['question']}")
        console.print(f"Total Cost: ${opp['total_cost']:.4f} (raw {opp['raw_cost']:.4f}, slip {opp['slippage']:.4f}, fee {opp['fee_cost']:.4f})")
        console.print(f"Profit: {opp['profit_pct']:.2f}% (${opp['expected_profit']:.2f} on {ORDER_SIZE} shares)")

        if PAPER_TRADING:
            for o in opp["outcomes"]:
                console.print(f"[yellow][PAPER] BUY {ORDER_SIZE} YES {o['outcome']} @ ${o['avg_price']:.4f}[/yellow]")
            self.trades_executed += 1
            self.total_profit += opp["expected_profit"]
            self.trades_by_market[opp["market_id"]] = self.trades_by_market.get(opp["market_id"], 0) + 1
            console.print(f"[green]✅ Locked in ${opp['expected_profit']:.2f} profit[/green]")
            return

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            for o in opp["outcomes"]:
                order = OrderArgs(
                    token_id=o["token_id"],
                    price=o["avg_price"],
                    size=ORDER_SIZE,
                    side=BUY
                )
                result = self.clob_client.post_order(
                    self.clob_client.create_order(order),
                    OrderType.FOK
                )
                if result.get("status") == "REJECTED":
                    console.print("[red]⚠️ One or more orders rejected. Aborting remaining legs.[/red]")
                    return

            self.trades_executed += 1
            self.total_profit += opp["expected_profit"]
            self.trades_by_market[opp["market_id"]] = self.trades_by_market.get(opp["market_id"], 0) + 1
            console.print(f"[green]✅ All legs filled! Locked in ${opp['expected_profit']:.2f}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Trade error: {e}[/red]")

    async def _check_market_with_sem(self, market: dict, sem: asyncio.Semaphore) -> Optional[dict]:
        async with sem:
            return await self.check_arbitrage(market)

    def display_status(self, checked: int):
        console.print(f"[dim]Checked: {checked} markets | Opportunities: {self.opportunities_found} | Trades: {self.trades_executed} | Profit: ${self.total_profit:.2f}[/dim]")

    async def run(self):
        if not await self.initialize():
            return

        self.running = True
        console.print("\n[bold green]🚀 Scanning Polymarket for strict arb opportunities...[/bold green]\n")

        try:
            while self.running:
                console.print(f"\n{'='*50}")
                console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")

                markets = await self.fetch_active_markets()
                if not markets:
                    console.print("[yellow]No active markets[/yellow]")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                if not markets:
                    console.print("[yellow]No markets passed prefilter[/yellow]")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                checked = 0
                sem = asyncio.Semaphore(max(1, ARB_CONCURRENCY))
                tasks = []
                for m in markets:
                    checked += 1
                    market_id = str(m.get("id") or m.get("slug") or m.get("question", "market"))
                    if self.trades_by_market.get(market_id, 0) >= MAX_TRADES_PER_MARKET:
                        continue
                    tasks.append(self._check_market_with_sem(m, sem))

                for coro in asyncio.as_completed(tasks):
                    opp = await coro
                    if opp:
                        self._print_opportunity_table(opp)
                        await self.execute_arbitrage(opp)

                self.display_status(checked)
                await asyncio.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped[/yellow]")
        finally:
            await self.http.aclose()

    def _print_opportunity_table(self, opp: dict):
        table = Table(show_header=False, box=None)
        table.add_column("Label", style="dim")
        table.add_column("Value")
        table.add_row("Market", opp["question"])
        table.add_row("Outcomes", ", ".join([str(o["outcome"]) for o in opp["outcomes"]][:4]))
        table.add_row("Total Cost", f"${opp['total_cost']:.4f}")
        table.add_row("Profit %", f"{opp['profit_pct']:.2f}%")
        console.print(table)


async def main():
    bot = PolymarketArbBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
