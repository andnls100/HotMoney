"""A-share hot-money/rally research MVP.

Downloads historical bars and Dragon-Tiger List records with AKShare, labels
rally events using a transparent forward-return rule, and exports pre-event
features for research. This is research tooling, not a trading recommendation.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd

LOG = logging.getLogger("hotmoney")

DEFAULT_SEATS = [
    "中国银河证券股份有限公司绍兴证券营业部",
    "国泰海通证券股份有限公司上海江苏路证券营业部",
    "中信证券股份有限公司西安朱雀大街证券营业部",
    "华鑫证券有限责任公司上海分公司",
    "东方财富证券股份有限公司拉萨团结路第二证券营业部",
    "东方财富证券股份有限公司拉萨东环路第二证券营业部",
]


def _normalise_code(value: object) -> str:
    code = "".join(ch for ch in str(value) if ch.isdigit())
    return code[-6:].zfill(6) if code else ""


def _date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def fetch_universe() -> pd.DataFrame:
    """Fetch current A-share symbols. Historical delisted coverage is not implied."""
    df = ak.stock_info_a_code_name()
    code_col = next((c for c in df.columns if str(c) in {"代码", "证券代码", "code"}), None)
    name_col = next((c for c in df.columns if str(c) in {"名称", "证券简称", "name"}), None)
    if not code_col or not name_col:
        raise RuntimeError(f"Unexpected AKShare universe schema: {list(df.columns)}")
    return df[[code_col, name_col]].rename(columns={code_col: "code", name_col: "name"}).assign(
        code=lambda x: x["code"].map(_normalise_code)
    ).drop_duplicates("code")


def fetch_bars(code: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / "bars" / f"{code}.csv"
    if cache.exists():
        old = pd.read_csv(cache, dtype={"代码": str, "code": str})
        if "日期" in old.columns and not old.empty:
            try:
                if pd.to_datetime(old["日期"]).min() <= pd.Timestamp(start) and pd.to_datetime(old["日期"]).max() >= pd.Timestamp(end):
                    return old
            except (ValueError, TypeError):
                pass
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=_date(start), end_date=_date(end), adjust=""
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "涨跌幅": "pct_change", "涨跌额": "change",
        "换手率": "turnover",
    })
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def fetch_lhb(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"lhb_{_date(start)}_{_date(end)}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"代码": str, "code": str})
    df = ak.stock_lhb_detail_em(start_date=_date(start), end_date=_date(end))
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        code_col = next((c for c in df.columns if c in {"代码", "证券代码"}), None)
        date_col = next((c for c in df.columns if c in {"上榜日", "日期"}), None)
        if code_col:
            df["code"] = df[code_col].map(_normalise_code)
        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def match_seats(code: str, date: pd.Timestamp, seats: Iterable[str], pause: float = 0.2) -> list[dict]:
    """Match configured seat names against a stock/date detail response.

    AKShare schemas vary; unrecognised schemas are logged and returned as no match.
    """
    try:
        detail = ak.stock_lhb_stock_detail_em(symbol=code, date=date.strftime("%Y%m%d"))
    except Exception as exc:  # API/data-source instability must not abort the full run.
        LOG.debug("LHB seat detail unavailable for %s %s: %s", code, date.date(), exc)
        time.sleep(pause)
        return []
    time.sleep(pause)
    if detail is None or detail.empty:
        return []
    seat_cols = [c for c in detail.columns if any(k in str(c) for k in ("营业部", "席位"))]
    found = []
    for _, row in detail.iterrows():
        for col in seat_cols:
            name = str(row.get(col, "")).strip()
            if name and name != "nan":
                for seat in seats:
                    if seat in name or name in seat:
                        found.append({"seat": seat, "reported_name": name, "column": str(col)})
    # Preserve evidence and remove duplicates; this is a seat-name match, not proof of identity.
    unique = {(x["seat"], x["reported_name"], x["column"]): x for x in found}
    return list(unique.values())


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    df = bars.copy().sort_values("date").reset_index(drop=True)
    for col in ("close", "open", "high", "low", "volume", "amount", "turnover", "pct_change"):
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
    # Event rule is deliberately explicit and configurable in spirit: >=15% close-to-close
    # gain over the next 3 sessions. Event day is the earliest day satisfying the rule.
    future_ret = df["close"].shift(-3) / df["close"] - 1
    df["rally_event"] = (future_ret >= 0.15).astype("Int64")
    df.loc[future_ret.isna(), "rally_event"] = pd.NA
    # Pre-event labels are retrospective labels for training, never live signals.
    df["pre_rally_1_5d"] = 0
    event_idx = df.index[df["rally_event"].eq(1).fillna(False)].tolist()
    for idx in event_idx:
        df.loc[max(0, idx - 5):idx - 1, "pre_rally_1_5d"] = 1
    return df


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    universe = fetch_universe()
    if args.symbols:
        wanted = {_normalise_code(x) for x in args.symbols.split(",")}
        universe = universe[universe["code"].isin(wanted)]
    if args.limit:
        universe = universe.head(args.limit)
    if universe.empty:
        raise SystemExit("No symbols selected.")
    LOG.info("Symbols selected: %d", len(universe))

    lhb = fetch_lhb(args.start, args.end, cache)
    if not lhb.empty:
        lhb.to_csv(out / "lhb_events.csv", index=False, encoding="utf-8-sig")
    seat_matches: list[dict] = []
    all_features: list[pd.DataFrame] = []
    for i, row in enumerate(universe.itertuples(index=False), 1):
        code, name = row.code, row.name
        try:
            bars = fetch_bars(code, args.start, args.end, cache)
            if bars.empty:
                continue
            features = build_features(bars)
            features["name"] = name
            all_features.append(features)
            # Only request seat details for dates where the symbol appears on LHB.
            if not lhb.empty and "code" in lhb and "date" in lhb:
                dates = lhb.loc[lhb["code"].eq(code), "date"].dropna().drop_duplicates()
                for date in dates:
                    for match in match_seats(code, pd.Timestamp(date), args.seats, args.pause):
                        seat_matches.append({"code": code, "name": name, "date": str(pd.Timestamp(date).date()), **match})
            if i % 100 == 0:
                LOG.info("Processed %d/%d", i, len(universe))
        except Exception as exc:
            LOG.warning("Skip %s %s: %s", code, name, exc)
    if all_features:
        result = pd.concat(all_features, ignore_index=True)
        result.to_csv(out / "daily_features.csv", index=False, encoding="utf-8-sig")
        events = result.loc[result["rally_event"].eq(1).fillna(False)].copy()
        events.to_csv(out / "rally_events.csv", index=False, encoding="utf-8-sig")
        pre = result.loc[result["pre_rally_1_5d"].eq(1)].copy()
        pre.to_csv(out / "pre_rally_samples.csv", index=False, encoding="utf-8-sig")
        cols = [c for c in ("ret_1d", "ret_3d", "ret_5d", "amount_ratio_5d", "volume_ratio_5d", "volatility_10d", "ma5_gap", "ma20_gap") if c in result]
        summary = {
            "date_range": [args.start, args.end],
            "symbols_processed": int(result["code"].nunique()),
            "rows": int(len(result)),
            "rally_events": int(len(events)),
            "pre_rally_samples": int(len(pre)),
            "pre_rally_feature_means": pre[cols].mean(numeric_only=True).replace({float("nan"): None}).to_dict() if not pre.empty else {},
            "control_feature_means": result.loc[result["rally_event"].eq(0).fillna(False), cols].mean(numeric_only=True).replace({float("nan"): None}).to_dict(),
            "event_rule": "close(t+3)/close(t)-1 >= 15%; retrospective label, not a live signal",
            "warnings": [
                "Universe uses current listed A-share symbols; delisted stocks may be missing.",
                "Seat names are heuristic matches and do not identify the actual trader.",
                "Pre-rally labels use future prices and must not be used as same-day predictors.",
                "Historical Dragon-Tiger List coverage depends on AKShare's upstream endpoint."
            ],
        }
        (out / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if seat_matches:
        pd.DataFrame(seat_matches).drop_duplicates().to_csv(out / "youzi_seat_matches.csv", index=False, encoding="utf-8-sig")
    LOG.info("Finished. Results: %s", out.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share rally event and hot-money seat research")
    parser.add_argument("--start", default=(pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", help="Comma-separated stock codes; omit to process current universe")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of symbols for a smoke test")
    parser.add_argument("--output", default="data", help="Output/cache directory")
    parser.add_argument("--pause", type=float, default=0.2, help="Delay between LHB detail requests")
    parser.add_argument("--seats", nargs="*", default=DEFAULT_SEATS, help="Seat-name keywords to match")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error("--start must be <= --end")
    run(args)


if __name__ == "__main__":
    main()
