"""
src/db.py
Database layer for SQLite recommendation history and lead CRM tracking.
Includes schema versioning, audit trails, and status definitions.
"""

import os
import sys
import sqlite3
import logging
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import MODEL_VERSION

logger = logging.getLogger("pricing_db")

CURRENT_SCHEMA_VERSION = 3

LEAD_STATUSES = [
    {"code": "Draft / Quoted", "label": "Draft / Quoted", "css": "draft"},
    {"code": "Negotiating", "label": "Negotiating", "css": "negotiating"},
    {"code": "Follow-up", "label": "Follow-up", "css": "followup"},
    {"code": "Under Review", "label": "Under Review", "css": "review"},
    {"code": "Won", "label": "Won", "css": "won"},
    {"code": "Lost", "label": "Lost", "css": "lost"}
]

VALID_STATUS_CODES = [s["code"] for s in LEAD_STATUSES]
STATUS_CSS_MAP = {s["code"]: s["css"] for s in LEAD_STATUSES}


def get_status_class(status: str) -> str:
    """Return badge CSS class for a given lead status."""
    return STATUS_CSS_MAP.get(status, "draft")


_INITIALIZED_DBS = set()


NEW_COLUMNS = {
    "final_price_inr": "REAL",
    "competitor_rate_per_kg": "REAL",
    "loss_reason_code": "TEXT"
}
LOSS_REASONS = ["price", "capacity", "schedule_transit", "customer_cancelled", "no_response", "other"]


def migrate_outcome_columns(db_path: str):
    """Ensure outcome tracking columns exist in the database."""
    abs_path = os.path.abspath(db_path)
    with sqlite3.connect(abs_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(recommendation_history)")
        have = {c[1] for c in cur.fetchall()}
        for col, typ in NEW_COLUMNS.items():
            if col not in have:
                cur.execute(f"ALTER TABLE recommendation_history ADD COLUMN {col} {typ}")
        conn.commit()


def record_outcome(
    db_path: str,
    quote_id: int,
    status: str,
    final_price_inr: Optional[float] = None,
    loss_reason: Optional[str] = None,
    competitor_rate_per_kg: Optional[float] = None
) -> bool:
    """
    Won -> requires final_price_inr (realized margin = final price - airline buy).
    Lost -> requires a coded loss_reason.
    """
    if status not in ("Won", "Lost"):
        return False
    if status == "Won" and (final_price_inr is None or final_price_inr <= 0):
        return False
    if status == "Lost" and loss_reason not in LOSS_REASONS:
        return False
    abs_path = os.path.abspath(db_path)
    migrate_outcome_columns(abs_path)
    with sqlite3.connect(abs_path, timeout=5.0) as conn:
        cur = conn.cursor()
        cur.execute("SELECT airline_buy_inr FROM recommendation_history WHERE id = ? AND is_deleted = 0", (quote_id,))
        row = cur.fetchone()
        if not row:
            return False
        realized = (final_price_inr - row[0]) if status == "Won" else None
        cur.execute("""UPDATE recommendation_history
                       SET status = ?, outcome_timestamp = CURRENT_TIMESTAMP, final_price_inr = ?,
                           realized_margin_inr = ?, loss_reason_code = ?, loss_reason = ?, competitor_rate_per_kg = ?
                       WHERE id = ?""", (status, final_price_inr, realized, loss_reason, loss_reason, competitor_rate_per_kg, quote_id))
        conn.commit()
        return cur.rowcount > 0


def get_recent_competitor_rate_benchmark(
    db_path: str,
    origin_port: str,
    dest_port: str,
    days: int = 45,
    max_records: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Retrieve recent competitor rate intelligence for a given origin-destination lane.
    Used by the pricing engine to calibrate margins against lost deals or known competitor pricing.
    """
    if not origin_port or not dest_port:
        return None
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(abs_path):
        return None
    migrate_outcome_columns(abs_path)

    with sqlite3.connect(abs_path, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        query = """
            SELECT competitor_rate_per_kg, timestamp, outcome_timestamp, status
            FROM recommendation_history
            WHERE origin_port = ? AND dest_port = ?
              AND competitor_rate_per_kg IS NOT NULL
              AND competitor_rate_per_kg > 0
              AND is_deleted = 0
              AND (timestamp >= datetime('now', ? || ' days') OR outcome_timestamp >= datetime('now', ? || ' days'))
            ORDER BY COALESCE(outcome_timestamp, timestamp) DESC
            LIMIT ?
        """
        cur.execute(query, (origin_port, dest_port, f"-{days}", f"-{days}", max_records))
        rows = cur.fetchall()

        if not rows:
            fallback_query = """
                SELECT competitor_rate_per_kg, timestamp, outcome_timestamp, status
                FROM recommendation_history
                WHERE origin_port = ? AND dest_port = ?
                  AND competitor_rate_per_kg IS NOT NULL
                  AND competitor_rate_per_kg > 0
                  AND is_deleted = 0
                ORDER BY COALESCE(outcome_timestamp, timestamp) DESC
                LIMIT ?
            """
            cur.execute(fallback_query, (origin_port, dest_port, max_records))
            rows = cur.fetchall()

        if not rows:
            return None

        rates = [float(r["competitor_rate_per_kg"]) for r in rows]
        if not rates:
            return None

        rates.sort()
        mid_idx = len(rates) // 2
        median_rate = rates[mid_idx] if len(rates) % 2 != 0 else (rates[mid_idx - 1] + rates[mid_idx]) / 2.0
        min_rate = min(rates)
        max_rate = max(rates)
        avg_rate = sum(rates) / len(rates)

        return {
            "origin_port": origin_port,
            "dest_port": dest_port,
            "sample_count": len(rates),
            "median_rate_per_kg": round(median_rate, 2),
            "min_rate_per_kg": round(min_rate, 2),
            "max_rate_per_kg": round(max_rate, 2),
            "avg_rate_per_kg": round(avg_rate, 2),
            "latest_timestamp": rows[0]["outcome_timestamp"] or rows[0]["timestamp"]
        }



def init_history_db(db_path: str = "data/recommendations_history.db", force: bool = False):
    """Initialize database, create performance indexes, and perform versioned schema migrations."""
    abs_path = os.path.abspath(db_path)
    if not force and abs_path in _INITIALIZED_DBS:
        return

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with sqlite3.connect(abs_path) as conn:
        cursor = conn.cursor()

        # Check existing user_version PRAGMA
        cursor.execute("PRAGMA user_version")
        ver_row = cursor.fetchone()
        version = ver_row[0] if ver_row else 0

        # Base table creation with full schema
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                origin_port TEXT,
                dest_port TEXT,
                airline_buy_inr REAL,
                selling_price_inr REAL,
                margin_pct REAL,
                margin_amount_inr REAL,
                win_prob REAL,
                expected_profit_inr REAL,
                gross_wt_kg REAL,
                chargeable_wt_kg REAL,
                commodity TEXT,
                client_type TEXT,
                branch TEXT,
                status TEXT DEFAULT 'Draft / Quoted',
                model_version TEXT DEFAULT '{MODEL_VERSION}',
                customer_name TEXT,
                inquiry_ref TEXT,
                chosen_strategy TEXT DEFAULT 'balanced',
                is_preset INTEGER DEFAULT 0,
                rate_per_kg REAL,
                cost_floor_price_inr REAL,
                max_margin_cap REAL,
                override_reason TEXT,
                outcome_timestamp DATETIME,
                realized_margin_inr REAL,
                loss_reason TEXT,
                loss_reason_code TEXT,
                final_price_inr REAL,
                competitor_rate_per_kg REAL,
                user_id TEXT,
                is_deleted INTEGER DEFAULT 0
            )
        """)

        # Migration logic
        cursor.execute("PRAGMA table_info(recommendation_history)")
        existing_cols = {c[1] for c in cursor.fetchall()}

        col_defs = {
            "status": "TEXT DEFAULT 'Draft / Quoted'",
            "model_version": f"TEXT DEFAULT '{MODEL_VERSION}'",
            "customer_name": "TEXT",
            "inquiry_ref": "TEXT",
            "chosen_strategy": "TEXT DEFAULT 'balanced'",
            "is_preset": "INTEGER DEFAULT 0",
            "rate_per_kg": "REAL",
            "cost_floor_price_inr": "REAL",
            "max_margin_cap": "REAL",
            "override_reason": "TEXT",
            "outcome_timestamp": "DATETIME",
            "realized_margin_inr": "REAL",
            "loss_reason": "TEXT",
            "loss_reason_code": "TEXT",
            "final_price_inr": "REAL",
            "competitor_rate_per_kg": "REAL",
            "user_id": "TEXT",
            "is_deleted": "INTEGER DEFAULT 0"
        }


        for col, col_type in col_defs.items():
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE recommendation_history ADD COLUMN {col} {col_type}")

        # Performance indexes (created after all columns exist)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_history_time ON recommendation_history(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_history_status ON recommendation_history(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_history_id_desc ON recommendation_history(id DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_history_active ON recommendation_history(is_deleted, is_preset)")

        cursor.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
        conn.commit()

    _INITIALIZED_DBS.add(abs_path)


def log_recommendation(
    db_path: str,
    payload: Dict[str, Any],
    optimal: Dict[str, Any],
    model_version: str = MODEL_VERSION,
    chosen_strategy: str = "balanced",
    is_preset: bool = False,
    override_reason: Optional[str] = None,
    user_id: Optional[str] = None
) -> bool:
    """
    Persist generated quote to SQLite audit history with enriched audit columns.
    """
    init_history_db(db_path)
    try:
        cust = str(payload.get("Customer_Company") or payload.get("customer_name") or "")
        inq_ref = str(payload.get("Inquiry Number") or payload.get("inquiry_ref") or "")
        strat = str(optimal.get("Strategy_Key") or chosen_strategy or "balanced")
        rate_kg = float(optimal["Rate_Per_Kg"]) if optimal.get("Rate_Per_Kg") is not None else 0.0
        floor_price = float(optimal["Cost_Floor_Price_INR"]) if optimal.get("Cost_Floor_Price_INR") is not None else 0.0
        max_margin_cap = float(optimal["Effective_Max_Margin_Cap"]) if optimal.get("Effective_Max_Margin_Cap") is not None else 0.0
        preset_flag = 1 if is_preset else 0

        win_prob = float(optimal["Win_Probability"]) if optimal.get("Win_Probability") is not None else None
        exp_profit = float(optimal["Expected_Profit_INR"]) if optimal.get("Expected_Profit_INR") is not None else None
        buy_inr = float(payload["Total_Buy_INR"]) if payload.get("Total_Buy_INR") is not None else 0.0
        sell_inr = float(optimal["Quoted_Sell_Price_INR"]) if optimal.get("Quoted_Sell_Price_INR") is not None else 0.0
        margin_pct = float(optimal["Margin_Percentage"]) if optimal.get("Margin_Percentage") is not None else 0.0
        margin_amt = float(optimal["Margin_Amount_INR"]) if optimal.get("Margin_Amount_INR") is not None else 0.0
        gross_wt = float(payload["Gross_Weight_Kg"]) if payload.get("Gross_Weight_Kg") is not None else 0.0
        chg_wt = float(payload["Chargeable_Weight_Kg"]) if payload.get("Chargeable_Weight_Kg") is not None else 0.0

        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recommendation_history (
                    origin_port, dest_port, airline_buy_inr, selling_price_inr,
                    margin_pct, margin_amount_inr, win_prob, expected_profit_inr,
                    gross_wt_kg, chargeable_wt_kg, commodity, client_type, branch,
                    status, model_version, customer_name, inquiry_ref,
                    chosen_strategy, is_preset, rate_per_kg, cost_floor_price_inr,
                    max_margin_cap, override_reason, user_id, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                str(payload.get("Origin_Port", "")),
                str(payload.get("Destination_Port", "")),
                buy_inr,
                sell_inr,
                margin_pct,
                margin_amt,
                win_prob,
                exp_profit,
                gross_wt,
                chg_wt,
                str(payload.get("Commodity_Group", "")),
                str(payload.get("Company_Group", "")),
                str(payload.get("Booked by branch", "")),
                "Draft / Quoted",
                str(model_version),
                cust,
                inq_ref,
                strat,
                preset_flag,
                rate_kg,
                floor_price,
                max_margin_cap,
                override_reason,
                user_id
            ))
            conn.commit()
            return True
    except Exception as err:
        logger.error(f"Audit History Logging Failed: {err}", exc_info=True)
        return False


def get_all_history(db_path: str) -> List[Dict[str, Any]]:
    """Retrieve active recommendation history records (excluding soft-deleted rows)."""
    init_history_db(db_path)
    records = []
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recommendation_history WHERE is_deleted = 0 ORDER BY id DESC")
        for row in cursor.fetchall():
            rec = dict(row)
            rec["status"] = rec.get("status") or "Draft / Quoted"
            rec["status_class"] = get_status_class(rec["status"])
            rec["model_version"] = rec.get("model_version") or MODEL_VERSION
            records.append(rec)
    return records


def get_history_stats(db_path: str) -> Dict[str, Any]:
    """
    Retrieve aggregate statistics across real commercial inquiries.
    Excludes test presets and soft-deleted records from KPI conversion metrics.
    """
    init_history_db(db_path)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*),
                SUM(CASE WHEN status = 'Won' THEN 1 ELSE 0 END),
                SUM(COALESCE(selling_price_inr, 0))
            FROM recommendation_history
            WHERE is_deleted = 0 AND is_preset = 0
        """)
        row = cursor.fetchone()
        total_count = row[0] if row and row[0] is not None else 0
        won_count = row[1] if row and row[1] is not None else 0
        total_pipeline = float(row[2]) if row and row[2] is not None else 0.0
        win_rate = (won_count / total_count * 100) if total_count > 0 else 0.0

        return {
            "total_count": total_count,
            "total_quotes": total_count,
            "won_count": won_count,
            "total_pipeline": total_pipeline,
            "win_rate": win_rate
        }


def get_history_paginated(
    db_path: str,
    page: int = 1,
    per_page: int = 15,
    include_presets: bool = False
) -> Dict[str, Any]:
    """Retrieve a paginated slice of active recommendation history records with consistent filtering."""
    init_history_db(db_path)
    migrate_outcome_columns(db_path)
    page = max(1, page)
    per_page = max(5, min(100, per_page))
    where = "is_deleted = 0" + ("" if include_presets else " AND is_preset = 0")

    with sqlite3.connect(db_path, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        total = cur.execute(f"SELECT COUNT(*) FROM recommendation_history WHERE {where}").fetchone()[0]
        pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, pages)
        off = (page - 1) * per_page
        recs = []
        for r in cur.execute(
            f"SELECT * FROM recommendation_history WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (per_page, off)
        ):
            d = dict(r)
            d["status"] = d.get("status") or "Draft / Quoted"
            d["status_class"] = get_status_class(d["status"])
            d["model_version"] = d.get("model_version") or MODEL_VERSION
            recs.append(d)

    return {
        "records": recs,
        "page": page,
        "per_page": per_page,
        "total_count": total,
        "total_pages": pages,
        "has_prev": page > 1,
        "has_next": page < pages,
        "prev_page": page - 1,
        "next_page": page + 1,
        "start_idx": off + 1 if total else 0,
        "end_idx": min(off + per_page, total),
        "stats": get_history_stats(db_path)
    }


def get_history_paginated_fixed(
    db_path: str,
    page: int = 1,
    per_page: int = 15,
    include_presets: bool = False
) -> Dict[str, Any]:
    return get_history_paginated(db_path, page, per_page, include_presets)


def update_lead_status(db_path: str, quote_id: int, new_status: str) -> bool:
    """Update commercial lifecycle status for a historical quote record."""
    status_clean = str(new_status).strip()
    if status_clean not in VALID_STATUS_CODES:
        norm_map = {
            "won": "Won",
            "won / booked": "Won",
            "won/booked": "Won",
            "lost": "Lost",
            "lost – won by competitor": "Lost",
            "lost - won by competitor": "Lost",
            "draft": "Draft / Quoted",
            "quoted": "Draft / Quoted",
            "draft / quoted": "Draft / Quoted",
            "draft/quoted": "Draft / Quoted",
            "negotiating": "Negotiating",
            "follow-up": "Follow-up",
            "followup": "Follow-up",
            "under review": "Under Review",
            "review": "Under Review"
        }
        status_clean = norm_map.get(status_clean.lower(), status_clean)

    if status_clean not in VALID_STATUS_CODES:
        return False
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE recommendation_history SET status = ? WHERE id = ?", (status_clean, quote_id))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as err:
        logger.error(f"Status update failed for ID {quote_id}: {err}", exc_info=True)
        return False


def clear_history_table(db_path: str, soft: bool = True) -> bool:
    """Clear or soft-delete records from recommendation history."""
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            if soft:
                cursor.execute("UPDATE recommendation_history SET is_deleted = 1")
            else:
                cursor.execute("DELETE FROM recommendation_history")
            conn.commit()
            return True
    except Exception as err:
        logger.error(f"Failed to clear history: {err}", exc_info=True)
        return False
