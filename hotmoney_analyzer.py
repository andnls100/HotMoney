"""A-share rally research using Tushare Pro.

Requires TUSHARE_TOKEN in the environment. Tushare's public interfaces provide
daily bars and Dragon-Tiger List summaries; they do not provide the broker-branch
seat-level detail needed to identify specific hot-money branches.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts

LOG = logging.getLogger("hotmoney")


def _normalise_code(value: object) -> str:
    value = str(value).strip()
    if "." in value:
        value = value.split(".", 1)[0]
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def _date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _pro():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing TUSHARE_TOKEN. Set your Tushare Pro token as an environment variable."
        )
    ts.set_token(token)
    return ts.pro_api(token)


def fetch_universe(pro) -> pd.DataFrame:
    """Fetch currently listed A-shares; delisted stocks are not included."""
    df = pro.stock_basic(
        exchange="", list_status="L", fields="ts_code,symbol,name,list_date"
    )
    if df is None or df.empty:
        raise RuntimeError("Tushare stock_basic returned no listed stocks.")
    return df.rename(columns={"symbol": "code"})[["code", "name", "ts_code", "list_date"]].assign(
        code=lambda x: x["code"].map(_normalise_code)
    ).drop_duplicates("code").reset_index(drop=True)


def fetch_bars(pro, code: str, ts_code: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / "bars" / f"{code}.csv"
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if cache.exists():
        old = pd.read_csv(cache, dtype={"code": str, "ts_code": str})
        if "date" in old.columns and not old.empty:
            dates = pd.to_datetime(old["date"], errors="coerce").dropna()
            if not dates.empty and dates.min() <= start_ts and dates.max() >= end_ts:
                return old

    raw = pro.daily(ts_code=ts_code, start_date=_date(start), end_date=_date(end))
    if raw is None or raw.empty:
        return pd.DataFrame()
    rename = {
        "trade_date": "date", "vol": "volume", "pct_chg": "pct_change",
    }
    df = raw.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["code"] = code
    df["ts_code"] = ts_code
    # Tushare daily amount is in thousand RMB; retain the source unit and document it.
    for col in ("open", "close", "high", "low", "volume", "amount", "pct_change", "change", "pre_close"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def fetch_lhb(pro, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"lhb_{_date(start)}_{_date(end)}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"ts_code": str, "code": str})
    # top_list provides stock-level Dragon-Tiger List summaries, not branch-level seats.
    df = pro.top_list(start_date=_date(start), end_date=_date(end))
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        df["code"] = df["ts_code"].map(_normalise_code)
        if "trade_date" in df:
            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def fetch_institutional_lhb(pro, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch Tushare institutional Dragon-Tiger details when the account has access."""
    cache = cache_dir / f"lhb_institutional_{_date(start)}_{_date(end)}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"ts_code": str, "code": str})
    try:
        df = pro.top_inst(start_date=_date(start), end_date=_date(end))
    except Exception as exc:
        LOG.warning("Tushare top_inst unavailable (permission/quota/API): %s", exc)
        df = pd.DataFrame()
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        df["code"] = df["ts_code"].map(_normalise_code)
        if "trade_date" in df:
            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    df = bars.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("close", "open", "high", "low", "volume", "amount", "pct_change"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ret_1d"] = df["close"].pct_change()
    for n in (3, 5, 10, 20):
        df[f"ret_{n}d"] = df["close"].pct_change(n)
    df["amount_ratio_5d"] = df["amount"] / df["amount"].rolling(5, min_periods=3).mean().shift(1)
    df["volume_ratio_5d"] = df["volume"] / df["volume"].rolling(5, min_periods=3).mean().shift(1)
    df["volatility_10d"] = df["ret_1d"].rolling(10, min_periods=5).std()
    df["ma5_gap"] = df["close"] / df["close"].rolling(5, min_periods=5).mean() - 1
    df["ma20_gap"] = df["close"] / df["close"].rolling(20, min_periods=10).mean() - 1
    future_ret = df["close"].shift(-3) / df["close"] - 1
    df["rally_event"] = (future_ret >= 0.15).astype("Int64")
    df.loc[future_ret.isna(), "rally_event"] = pd.NA
    df["pre_rally_1_5d"] = 0
    event_idx = df.index[df["rally_event"].eq(1).fillna(False)].tolist()
    for idx in event_idx:
        if idx > 0:
            df.loc[max(0, idx - 5):idx - 1, "pre_rally_1_5d"] = 1
    return df


def run(args: argparse.Namespace) -> None:
    pro = _pro()
    out = Path(args.output)
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe(pro)
    if args.symbols:
        wanted = {_normalise_code(x) for x in args.symbols.split(",")}
        universe = universe[universe["code"].isin(wanted)]
    if args.limit:
        universe = universe.head(args.limit)
    if universe.empty:
        raise SystemExit("No symbols selected.")
    LOG.info("Symbols selected: %d", len(universe))

    lhb = fetch_lhb(pro, args.start, args.end, cache)
    lhb.to_csv(out / "lhb_events.csv", index=False, encoding="utf-8-sig")
    institutional = fetch_institutional_lhb(pro, args.start, args.end, cache)
    institutional.to_csv(out / "lhb_institutional_events.csv", index=False, encoding="utf-8-sig")

    all_features: list[pd.DataFrame] = []
    for i, row in enumerate(universe.itertuples(index=False), 1):
        try:
            bars = fetch_bars(pro, row.code, row.ts_code, args.start, args.end, cache)
            if bars.empty:
                LOG.debug("No bars for %s", row.ts_code)
                continue
            features = build_features(bars)
            features["name"] = row.name
            all_features.append(features)
            if args.pause > 0:
                time.sleep(args.pause)
            if i % 100 == 0:
                LOG.info("Processed %d/%d", i, len(universe))
        except Exception as exc:
            LOG.warning("Skip %s %s: %s", row.code, row.name, exc)

    if all_features:
        result = pd.concat(all_features, ignore_index=True)
        result.to_csv(out / "daily_features.csv", index=False, encoding="utf-8-sig")
        events = result.loc[result["rally_event"].eq(1).fillna(False)].copy()
        events.to_csv(out / "rally_events.csv", index=False, encoding="utf-8-sig")
        pre = result.loc[result["pre_rally_1_5d"].eq(1)].copy()
        pre.to_csv(out / "pre_rally_samples.csv", index=False, encoding="utf-8-sig")
        cols = [c for c in ("ret_1d", "ret_3d", "ret_5d", "amount_ratio_5d", "volume_ratio_5d", "volatility_10d", "ma5_gap", "ma20_gap") if c in result]
        control = result.loc[result["rally_event"].eq(0).fillna(False), cols]
        summary = {
            "data_source": "Tushare Pro",
            "date_range": [args.start, args.end],
            "symbols_processed": int(result["code"].nunique()),
            "rows": int(len(result)),
            "rally_events": int(len(events)),
            "pre_rally_samples": int(len(pre)),
            "pre_rally_feature_means": pre[cols].mean(numeric_only=True).replace({float("nan"): None}).to_dict() if not pre.empty else {},
            "control_feature_means": control.mean(numeric_only=True).replace({float("nan"): None}).to_dict(),
            "event_rule": "close(t+3)/close(t)-1 >= 15%; retrospective label, not a live signal",
            "warnings": [
                "Universe uses currently listed A-shares; delisted stocks may be missing.",
                "Tushare top_list is stock-level LHB summary and does not contain broker-branch seat names.",
                "Institutional LHB details require relevant Tushare Pro permissions and may be unavailable.",
                "Pre-rally labels use future prices and must not be used as same-day predictors.",
                "Tushare API permissions, quotas, and point limits depend on the account.",
                "Tushare daily amount is in thousand RMB; volume is in lots."
            ],
        }
        (out / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Finished. Results: %s", out.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share rally event research using Tushare Pro")
    parser.add_argument("--start", default=(pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", help="Comma-separated stock codes; omit to process current universe")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum symbols for a smoke test")
    parser.add_argument("--output", default="data", help="Output/cache directory")
    parser.add_argument("--pause", type=float, default=0.05, help="Delay between per-stock API requests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error("--start must be <= --end")
    run(args)


if __name__ == "__main__":
    main()
