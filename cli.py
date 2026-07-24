"""Command-line entry point for the optvol pricing/divergence engine.

Examples
--------
    python cli.py AAPL
    python cli.py SPY --expiry 2026-07-17
    python cli.py NVDA --expiries        # list available expiries
"""
from __future__ import annotations

import argparse

from optvol.analyze import enrich_chain, summary
from optvol.data import list_expiries, load_chain


def main():
    parser = argparse.ArgumentParser(
        description="Solve implied vols and surface model-vs-market divergence "
        "for a live option chain."
    )
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--expiry", default=None,
        help="Expiry in YYYY-MM-DD (defaults to nearest).",
    )
    parser.add_argument(
        "--expiries", action="store_true",
        help="List available expiries for the symbol and exit.",
    )
    parser.add_argument(
        "--show-drops", action="store_true",
        help="Print the data-cleaning drop report.",
    )
    args = parser.parse_args()

    if args.expiries:
        print("\n".join(list_expiries(args.symbol)))
        return

    snap = load_chain(args.symbol, args.expiry)
    df = enrich_chain(snap)
    print(summary(df, snap))

    if args.show_drops:
        print("\nData-cleaning drop report:")
        for k, v in snap.drop_report.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
