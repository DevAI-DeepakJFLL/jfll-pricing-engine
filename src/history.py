"""
src/history.py
Point-in-time customer history and relationship metrics.
Computes metrics strictly from earlier quotes within a rolling 365-day window.
"""

from typing import Optional
import numpy as np
import pandas as pd


def add_customer_history(
    df: pd.DataFrame,
    date_col: str = "quote_date",
    cust_col: str = "Customer_Company"
) -> pd.DataFrame:
    """
    Compute point-in-time customer historical metrics strictly from quotes prior to quote_date.
    Features added:
      - cust_quotes_365d: Number of quotes for customer in rolling 365 days
      - cust_win_rate_365d: Win rate in rolling 365 days
      - cust_last_won_mpk: Last won margin INR/kg
      - cust_median_won_mpk_365d: Median won margin INR/kg in rolling 365 days
    """
    if date_col not in df.columns:
        if "Year_Month" in df.columns:
            # Approximate date from Year_Month if explicit date missing
            d = df.copy()
            d[date_col] = pd.to_datetime(d["Year_Month"] + "-01")
        else:
            return df.copy()
    else:
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col])

    d = d.sort_values(date_col).copy()
    d["_won"] = (d["is_won"] == 1).astype(int) if "is_won" in d.columns else 0
    d["_wmpk"] = np.where(
        d["_won"] == 1,
        d["Margin_Amount_INR"] / d["Chargeable_Weight_Kg"].replace(0, np.nan),
        np.nan
    ) if "Margin_Amount_INR" in d.columns and "Chargeable_Weight_Kg" in d.columns else np.nan

    out = []
    for _, g in d.groupby(cust_col, sort=False):
        g = g.set_index(date_col)
        prior_n = g["_won"].shift(1).rolling("365D", min_periods=1).count().fillna(0)
        prior_w = g["_won"].shift(1).rolling("365D", min_periods=1).sum().fillna(0)
        g = g.assign(
            cust_quotes_365d=prior_n,
            cust_win_rate_365d=(prior_w / prior_n.replace(0, np.nan)),
            cust_last_won_mpk=g["_wmpk"].shift(1).ffill(),
            cust_median_won_mpk_365d=g["_wmpk"].shift(1).rolling("365D", min_periods=1).median()
        )
        out.append(g.reset_index())

    res = pd.concat(out).drop(columns=["_won", "_wmpk"]).sort_values(date_col)
    return res
