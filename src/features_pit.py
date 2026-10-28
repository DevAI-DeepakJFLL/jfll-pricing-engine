"""
src/features_pit.py
Point-in-Time Rolling Customer & Lane Features.

Computes point-in-time windowed features strictly excluding the current quote and
any future transactions (t < T, zero lookahead leakage, resolving Issue D5 & P2).

Provides:
  - compute_point_in_time_features: Batch computation on historical DataFrames.
  - CustomerPITRegistry: Rolling in-memory registry for streaming inference / real-time serving.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Any, Union
import bisect
import numpy as np
import pandas as pd


# Default prior constants
DEFAULT_RECENCY_DAYS: float = 999.0
DEFAULT_WIN_RATE: float = 0.50
LOOKBACK_CUSTOMER_DAYS: int = 180
LOOKBACK_LANE_DAYS: int = 90


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from candidates present in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_schema(
    df: pd.DataFrame,
    date_col: str,
    cust_col: str,
    lane_col: str,
) -> Dict[str, Optional[str]]:
    """Resolve column names for date, customer, lane, outcome, margin, and sell INR."""
    # Date
    resolved_date = date_col if date_col in df.columns else _detect_column(
        df, ["quote_date", "Year_Month", "date", "timestamp", "Quote_Date"]
    )
    if resolved_date is None:
        raise ValueError(
            f"Date column '{date_col}' not found and no suitable date column detected."
        )

    # Customer
    resolved_cust = cust_col if cust_col in df.columns else _detect_column(
        df, ["Customer_Company", "customer_id", "Customer", "account_id", "Company"]
    )
    if resolved_cust is None:
        raise ValueError(
            f"Customer column '{cust_col}' not found and no suitable customer column detected."
        )

    # Lane
    resolved_lane = lane_col if lane_col in df.columns else _detect_column(
        df, ["Regional_Lane", "regional_lane", "Trade_Lane", "lane", "lane_name"]
    )

    # Outcome (is_won)
    resolved_won = _detect_column(df, ["is_won", "won_flag", "Quote Status", "Inquiry Status"])

    # Margin
    resolved_margin = _detect_column(
        df, ["Margin_Percentage", "margin", "Margin_Pct", "margin_percentage", "Actual_Margin"]
    )

    # Sell INR
    resolved_sell = _detect_column(
        df, ["Total_Sell_INR", "sell_inr", "revenue", "Total_Sell", "sell", "Sell_INR"]
    )

    return {
        "date": resolved_date,
        "cust": resolved_cust,
        "lane": resolved_lane,
        "won": resolved_won,
        "margin": resolved_margin,
        "sell": resolved_sell,
    }


def compute_point_in_time_features(
    df: pd.DataFrame,
    date_col: str = "quote_date",
    cust_col: str = "Customer_Company",
    lane_col: str = "Regional_Lane",
) -> pd.DataFrame:
    """
    Compute point-in-time historical features strictly from transactions occurring
    prior to each quote's date (t < T, zero lookahead leakage).

    Features generated per quote row:
      1. cust_pit_won_count_180d: Total completed Won shipments for customer in trailing 180 days.
      2. cust_pit_inquiry_count_180d: Total inquiries for customer in trailing 180 days.
      3. cust_pit_recency_days: Calendar days elapsed since customer's most recent prior booking (if none, 999.0).
      4. cust_pit_median_margin_won: Median margin percentage on customer's prior won deals (if none, NaN).
      5. cust_pit_total_spend_inr_180d: Cumulative won spend in trailing 180 days.
      6. cust_pit_win_rate_180d: cust_pit_won_count_180d / cust_pit_inquiry_count_180d (if no prior inquiries, 0.50).
      7. lane_pit_median_margin_won_90d: Median won margin on regional lane in trailing 90 days prior to quote date.

    Parameters:
      df: Input quotations DataFrame.
      date_col: Name of quote date column (defaults to 'quote_date').
      cust_col: Name of customer identifier column (defaults to 'Customer_Company').
      lane_col: Name of regional lane column (defaults to 'Regional_Lane').

    Returns:
      DataFrame containing all original columns plus the 7 point-in-time features,
      indexed identically to the input DataFrame.
    """
    if df.empty:
        res = df.copy()
        for col, default_val in [
            ("cust_pit_won_count_180d", 0),
            ("cust_pit_inquiry_count_180d", 0),
            ("cust_pit_recency_days", DEFAULT_RECENCY_DAYS),
            ("cust_pit_median_margin_won", np.nan),
            ("cust_pit_total_spend_inr_180d", 0.0),
            ("cust_pit_win_rate_180d", DEFAULT_WIN_RATE),
            ("lane_pit_median_margin_won_90d", np.nan),
        ]:
            res[col] = pd.Series(dtype=type(default_val))
        return res

    schema = _resolve_schema(df, date_col, cust_col, lane_col)

    # Work on a copy with parsed standard working columns
    work_df = df.copy()

    # Parse dates
    date_src = schema["date"]
    if date_src == "Year_Month" and "Year_Month" in work_df.columns:
        dt_series = pd.to_datetime(work_df["Year_Month"].astype(str) + "-01")
    else:
        dt_series = pd.to_datetime(work_df[date_src])
    work_df["_pit_dt"] = dt_series

    # Parse outcome (is_won)
    won_col = schema["won"]
    if won_col is not None:
        if won_col in ["Quote Status", "Inquiry Status"]:
            work_df["_pit_won"] = (
                work_df[won_col].astype(str).str.strip().str.lower() == "won"
            ).astype(int)
        else:
            work_df["_pit_won"] = (
                pd.to_numeric(work_df[won_col], errors="coerce").fillna(0) == 1
            ).astype(int)
    else:
        work_df["_pit_won"] = 0

    # Parse margin
    margin_col = schema["margin"]
    if margin_col is not None:
        work_df["_pit_margin"] = pd.to_numeric(work_df[margin_col], errors="coerce")
    else:
        work_df["_pit_margin"] = np.nan

    # Parse sell INR
    sell_col = schema["sell"]
    if sell_col is not None:
        work_df["_pit_sell"] = pd.to_numeric(work_df[sell_col], errors="coerce").fillna(0.0)
    else:
        work_df["_pit_sell"] = 0.0

    cust_key = schema["cust"]
    lane_key = schema["lane"]

    # Preallocate output arrays
    n = len(work_df)
    cust_won_count_180d = np.zeros(n, dtype=int)
    cust_inquiry_count_180d = np.zeros(n, dtype=int)
    cust_recency_days = np.full(n, DEFAULT_RECENCY_DAYS, dtype=float)
    cust_median_margin_won = np.full(n, np.nan, dtype=float)
    cust_total_spend_180d = np.zeros(n, dtype=float)
    cust_win_rate_180d = np.full(n, DEFAULT_WIN_RATE, dtype=float)
    lane_median_margin_90d = np.full(n, np.nan, dtype=float)

    work_df["_orig_pos"] = np.arange(n)

    # 1. Customer features computation
    # Group row positions by customer
    for cust_id, group in work_df.groupby(cust_key, sort=False):
        group_pos = group["_orig_pos"].values
        # Sort group by datetime stably
        sort_order = np.argsort(group["_pit_dt"].values, kind="mergesort")
        sorted_pos = group_pos[sort_order]

        cust_dts = group["_pit_dt"].values[sort_order]
        cust_wons = group["_pit_won"].values[sort_order]
        cust_margins = group["_pit_margin"].values[sort_order]
        cust_sells = group["_pit_sell"].values[sort_order]

        # Precompute won sub-indices for fast recency and margin lookup
        won_mask = cust_wons == 1
        won_dts = cust_dts[won_mask]
        won_margins = cust_margins[won_mask]

        delta_180 = np.timedelta64(LOOKBACK_CUSTOMER_DAYS, "D")

        # For each row in this customer group
        for pos_idx, row_pos in enumerate(sorted_pos):
            current_t = cust_dts[pos_idx]

            # Binary search strictly prior transactions: t < current_t
            # side='left' strictly excludes any transaction with t >= current_t
            idx_right = np.searchsorted(cust_dts, current_t, side="left")

            if idx_right == 0:
                # No prior history for this customer
                continue

            # Prior transactions strictly before current_t
            idx_left_180 = np.searchsorted(cust_dts, current_t - delta_180, side="left")

            inquiries_180 = idx_right - idx_left_180
            cust_inquiry_count_180d[row_pos] = inquiries_180

            # 180d window slices
            wons_in_window = cust_wons[idx_left_180:idx_right]
            sells_in_window = cust_sells[idx_left_180:idx_right]

            won_cnt_180 = int(np.sum(wons_in_window))
            cust_won_count_180d[row_pos] = won_cnt_180

            spend_180 = float(np.sum(sells_in_window * wons_in_window))
            cust_total_spend_180d[row_pos] = spend_180

            if inquiries_180 > 0:
                cust_win_rate_180d[row_pos] = float(won_cnt_180) / float(inquiries_180)
            else:
                cust_win_rate_180d[row_pos] = DEFAULT_WIN_RATE

            # Recency and median margin on all prior won deals
            idx_won_right = np.searchsorted(won_dts, current_t, side="left")
            if idx_won_right > 0:
                most_recent_won_dt = won_dts[idx_won_right - 1]
                # Calendar days elapsed
                recency_ns = current_t - most_recent_won_dt
                recency_val = float(recency_ns / np.timedelta64(1, "D"))
                cust_recency_days[row_pos] = max(recency_val, 0.0)

                prior_won_margins = won_margins[:idx_won_right]
                valid_margins = prior_won_margins[~np.isnan(prior_won_margins)]
                if len(valid_margins) > 0:
                    cust_median_margin_won[row_pos] = float(np.median(valid_margins))

    # 2. Lane features computation
    if lane_key is not None and lane_key in work_df.columns:
        delta_90 = np.timedelta64(LOOKBACK_LANE_DAYS, "D")
        for lane_id, group in work_df.groupby(lane_key, sort=False):
            # Only won deals with valid margins contribute to lane margin benchmark
            won_lane = group[group["_pit_won"] == 1]
            if won_lane.empty:
                continue

            sort_order_won = np.argsort(won_lane["_pit_dt"].values, kind="mergesort")
            lane_won_dts = won_lane["_pit_dt"].values[sort_order_won]
            lane_won_margins = won_lane["_pit_margin"].values[sort_order_won]

            group_pos_lane = group["_orig_pos"].values
            group_dts_lane = group["_pit_dt"].values
            for i, row_pos in enumerate(group_pos_lane):
                current_t = group_dts_lane[i]
                idx_right_lane = np.searchsorted(lane_won_dts, current_t, side="left")
                if idx_right_lane == 0:
                    continue

                idx_left_90 = np.searchsorted(lane_won_dts, current_t - delta_90, side="left")
                window_margins = lane_won_margins[idx_left_90:idx_right_lane]
                valid_lane_margins = window_margins[~np.isnan(window_margins)]
                if len(valid_lane_margins) > 0:
                    lane_median_margin_90d[row_pos] = float(np.median(valid_lane_margins))

    # Construct result dataframe
    out_df = df.copy()
    out_df["cust_pit_won_count_180d"] = cust_won_count_180d
    out_df["cust_pit_inquiry_count_180d"] = cust_inquiry_count_180d
    out_df["cust_pit_recency_days"] = cust_recency_days
    out_df["cust_pit_median_margin_won"] = cust_median_margin_won
    out_df["cust_pit_total_spend_inr_180d"] = cust_total_spend_180d
    out_df["cust_pit_win_rate_180d"] = cust_win_rate_180d
    out_df["lane_pit_median_margin_won_90d"] = lane_median_margin_90d

    return out_df


class CustomerPITRegistry:
    """
    In-memory stateful registry maintaining rolling point-in-time metrics
    for real-time serving / streaming inference.
    """

    def __init__(self):
        # Maps customer_id -> sorted list of tuples: (timestamp, is_won, margin, sell_inr)
        self._customers: Dict[str, List[tuple]] = defaultdict(list)
        # Maps regional_lane -> sorted list of tuples: (timestamp, won_margin)
        self._lanes: Dict[str, List[tuple]] = defaultdict(list)

    def update(
        self,
        quote_date: Any,
        customer_id: str,
        is_won: Union[int, bool, str],
        margin: Optional[float] = None,
        sell_inr: Optional[float] = 0.0,
        regional_lane: Optional[str] = None,
        lane: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Record a completed transaction into the streaming registry.

        Parameters:
          quote_date: Transaction date (string or datetime).
          customer_id: Canonical customer identifier.
          is_won: Won status (1/0, True/False, 'Won'/'Lost').
          margin: Realized margin percentage (optional).
          sell_inr: Realized sell INR amount (optional).
          regional_lane: Regional trade lane identifier (optional).
          lane: Alias for regional_lane.
        """
        dt = pd.to_datetime(quote_date)

        # Normalize is_won
        if isinstance(is_won, str):
            won_int = 1 if is_won.strip().lower() == "won" else 0
        else:
            won_int = 1 if int(is_won) == 1 else 0

        # Normalize margin
        margin_val = float(margin) if margin is not None and not np.isnan(margin) else np.nan

        # Normalize sell_inr
        sell_val = float(sell_inr) if sell_inr is not None and not np.isnan(sell_inr) else 0.0

        resolved_lane = regional_lane if regional_lane is not None else lane

        # Insert customer record maintaining timestamp sort order
        cust_list = self._customers[str(customer_id)]
        cust_entry = (dt, won_int, margin_val, sell_val)
        # Use bisect on timestamp with O(1) allocation key
        ins_idx = bisect.bisect_right(cust_list, dt, key=lambda entry: entry[0])
        cust_list.insert(ins_idx, cust_entry)

        # Insert lane record if won with valid margin
        if won_int == 1 and not np.isnan(margin_val) and resolved_lane:
            lane_list = self._lanes[str(resolved_lane)]
            lane_entry = (dt, margin_val)
            lane_ins_idx = bisect.bisect_right(lane_list, dt, key=lambda entry: entry[0])
            lane_list.insert(lane_ins_idx, lane_entry)

    def get_features(
        self,
        quote_date: Any,
        customer_id: str,
        regional_lane: Optional[str] = None,
        lane: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Query point-in-time features strictly prior to the specified quote_date (t < T).

        Returns:
          Dict containing the 7 point-in-time customer and lane features.
        """
        current_t = pd.to_datetime(quote_date)
        resolved_lane = regional_lane if regional_lane is not None else lane

        cust_key_str = str(customer_id)
        cust_list = self._customers.get(cust_key_str)
        if not cust_list:
            # Fallback 1: Resolve canonical slug if raw company name was queried
            try:
                from src.customer_normalizer import get_canonical_customer_id
            except ImportError:
                try:
                    from customer_normalizer import get_canonical_customer_id
                except ImportError:
                    get_canonical_customer_id = None
            if get_canonical_customer_id is not None:
                canonical_id = get_canonical_customer_id(cust_key_str)
                cust_list = self._customers.get(canonical_id)

        if not cust_list and cust_key_str.startswith("CUST_"):
            # Fallback 2: Check raw/normalized company name without CUST_ prefix
            stripped = cust_key_str[5:].replace("_", " ")
            cust_list = self._customers.get(stripped)

        if not cust_list:
            cust_list = []

        # Default outputs
        won_count_180d = 0
        inquiry_count_180d = 0
        recency_days = DEFAULT_RECENCY_DAYS
        median_margin_won = np.nan
        total_spend_180d = 0.0
        win_rate_180d = DEFAULT_WIN_RATE
        lane_median_margin_90d = np.nan

        if cust_list:
            # Strict point-in-time: transactions where t < current_t using zero-allocation bisect key
            idx_right = bisect.bisect_left(cust_list, current_t, key=lambda entry: entry[0])

            if idx_right > 0:
                t_180 = current_t - pd.Timedelta(days=LOOKBACK_CUSTOMER_DAYS)
                idx_left_180 = bisect.bisect_left(cust_list, t_180, key=lambda entry: entry[0])

                prior_slice = cust_list[:idx_right]
                window_slice = cust_list[idx_left_180:idx_right]

                inquiry_count_180d = len(window_slice)
                won_entries_180 = [e for e in window_slice if e[1] == 1]
                won_count_180d = len(won_entries_180)
                total_spend_180d = float(sum(e[3] for e in won_entries_180))

                if inquiry_count_180d > 0:
                    win_rate_180d = float(won_count_180d) / float(inquiry_count_180d)
                else:
                    win_rate_180d = DEFAULT_WIN_RATE

                # Lifetime prior won deals strictly before current_t
                all_prior_wons = [e for e in prior_slice if e[1] == 1]
                if all_prior_wons:
                    # Most recent won booking
                    last_won_dt = all_prior_wons[-1][0]
                    recency_delta = (current_t - last_won_dt) / pd.Timedelta(days=1)
                    recency_days = max(float(recency_delta), 0.0)

                    valid_margins = [
                        e[2] for e in all_prior_wons if not np.isnan(e[2])
                    ]
                    if valid_margins:
                        median_margin_won = float(np.median(valid_margins))

        # Lane 90d won margin
        if resolved_lane and str(resolved_lane) in self._lanes:
            lane_list = self._lanes[str(resolved_lane)]
            idx_right_lane = bisect.bisect_left(lane_list, current_t, key=lambda entry: entry[0])

            if idx_right_lane > 0:
                t_90 = current_t - pd.Timedelta(days=LOOKBACK_LANE_DAYS)
                idx_left_90 = bisect.bisect_left(lane_list, t_90, key=lambda entry: entry[0])
                lane_window = lane_list[idx_left_90:idx_right_lane]
                lane_margins = [e[1] for e in lane_window if not np.isnan(e[1])]
                if lane_margins:
                    lane_median_margin_90d = float(np.median(lane_margins))

        return {
            "cust_pit_won_count_180d": won_count_180d,
            "cust_pit_inquiry_count_180d": inquiry_count_180d,
            "cust_pit_recency_days": recency_days,
            "cust_pit_median_margin_won": median_margin_won,
            "cust_pit_total_spend_inr_180d": total_spend_180d,
            "cust_pit_win_rate_180d": win_rate_180d,
            "lane_pit_median_margin_won_90d": lane_median_margin_90d,
        }

    def load_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str = "quote_date",
        cust_col: str = "Customer_Company",
        lane_col: str = "Regional_Lane",
    ) -> "CustomerPITRegistry":
        """Load history from a DataFrame into the registry."""
        if df.empty:
            return self

        schema = _resolve_schema(df, date_col, cust_col, lane_col)
        date_src = schema["date"]
        cust_key = schema["cust"]
        lane_key = schema["lane"]
        won_col = schema["won"]
        margin_col = schema["margin"]
        sell_col = schema["sell"]

        date_vals = df[date_src].values
        cust_vals = df[cust_key].values
        lane_vals = df[lane_key].values if lane_key and lane_key in df.columns else [None] * len(df)
        won_vals = df[won_col].values if won_col and won_col in df.columns else [0] * len(df)
        margin_vals = df[margin_col].values if margin_col and margin_col in df.columns else [np.nan] * len(df)
        sell_vals = df[sell_col].values if sell_col and sell_col in df.columns else [0.0] * len(df)

        for qd, cid, lval, wval, mval, sval in zip(
            date_vals, cust_vals, lane_vals, won_vals, margin_vals, sell_vals
        ):
            self.update(
                quote_date=qd,
                customer_id=cid,
                is_won=wval,
                margin=mval,
                sell_inr=sval,
                regional_lane=lval,
            )
        return self

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        date_col: str = "quote_date",
        cust_col: str = "Customer_Company",
        lane_col: str = "Regional_Lane",
    ) -> "CustomerPITRegistry":
        """Instantiate and populate registry from a DataFrame."""
        registry = cls()
        registry.load_dataframe(
            df=df,
            date_col=date_col,
            cust_col=cust_col,
            lane_col=lane_col,
        )
        return registry
