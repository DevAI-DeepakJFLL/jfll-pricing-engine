#!/usr/bin/env python3
"""
Generate Deepak_-_Executive_Work_Status_Tracker_and_Gantt_Chart.xlsx
Updated version:
1. Executive Summary sheet is completely removed.
2. Gantt Chart sheet groups all related tasks by project and work category with:
   - Top KPI metric ribbon
   - Collapsible project summary rows with rollups
   - Clear section dividers
   - Indented child task rows
   - Dynamic formulas for duration, start, end, and progress
   - Visual week-by-week Gantt matrix
   - Embedded native Excel BarChart
3. Detailed sheets (Work with Tyrus, Work with Tyra, Dependencies & Vendors) preserved.
"""

import os
import sys
from datetime import date, datetime, timedelta
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference, Series
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.fill import ColorChoice, PatternFillProperties

OUTPUT_PATH = '/Users/deepak/Downloads/Deepak_-_Executive_Work_Status_Tracker_and_Gantt_Chart.xlsx'
WORKSPACE_PATH = '/Users/deepak/Downloads/air_export_pricing_engine/Deepak_-_Executive_Work_Status_Tracker_and_Gantt_Chart.xlsx'

# Brand & UI Palette
FONT_FAMILY = "Segoe UI"
COLOR_NAVY = "0F172A"       # Primary Dark Slate
COLOR_NAVY_LIGHT = "1E293B" # Dark Slate lighter
COLOR_CYAN = "23C2F2"       # Primary Cyan Accent
COLOR_CYAN_DARK = "0EA8DA"  # Primary Dark Cyan
COLOR_LIME = "A7CF45"       # Secondary Lime Green
COLOR_LIME_DARK = "84A82F"
COLOR_BG_LIGHT = "F8FAFC"   # Slate 50
COLOR_BG_MUTED = "F1F5F9"   # Slate 100
COLOR_CARD_BORDER = "CBD5E1"# Slate 300
COLOR_BORDER_LIGHT = "E2E8F0"
COLOR_WHITE = "FFFFFF"
COLOR_MUTED_TEXT = "64748B"
COLOR_DARK_TEXT = "0F172A"

# Status styles: (badge fill, badge font, Gantt bar fill)
STATUS_STYLES = {
    "Completed": {
        "fill": "D1FAE5",      # Emerald 100
        "font": "065F46",      # Emerald 800
        "bar_fill": "10B981",  # Emerald 500
        "badge": "Completed"
    },
    "In Progress": {
        "fill": "FEF3C7",      # Amber 100
        "font": "92400E",      # Amber 800
        "bar_fill": "F59E0B",  # Amber 500
        "badge": "In Progress"
    },
    "Waiting on Dependency": {
        "fill": "FEE2E2",      # Rose 100
        "font": "991B1B",      # Rose 800
        "bar_fill": "EF4444",  # Rose 500
        "badge": "Waiting on Dependency"
    },
    "Needs Decision": {
        "fill": "EDE9FE",      # Violet 100
        "font": "5B21B6",      # Violet 800
        "bar_fill": "8B5CF6",  # Violet 500
        "badge": "Needs Decision"
    },
    "Planned": {
        "fill": "E0E7FF",      # Indigo 100
        "font": "3730A3",      # Indigo 800
        "bar_fill": "6366F1",  # Indigo 500
        "badge": "Planned / Ready"
    }
}

# Borders
border_thin = Border(
    left=Side(style='thin', color=COLOR_BORDER_LIGHT),
    right=Side(style='thin', color=COLOR_BORDER_LIGHT),
    top=Side(style='thin', color=COLOR_BORDER_LIGHT),
    bottom=Side(style='thin', color=COLOR_BORDER_LIGHT)
)

border_header = Border(
    left=Side(style='thin', color=COLOR_NAVY_LIGHT),
    right=Side(style='thin', color=COLOR_NAVY_LIGHT),
    top=Side(style='thin', color=COLOR_NAVY_LIGHT),
    bottom=Side(style='medium', color=COLOR_CYAN)
)

# 11 Calendar Weeks (Sep 01 - Nov 16, 2026)
GANTT_WEEKS = [
    ("W36 (01-07 Sep)", date(2026, 9, 1), date(2026, 9, 7)),
    ("W37 (08-14 Sep)", date(2026, 9, 8), date(2026, 9, 14)),
    ("W38 (15-21 Sep)", date(2026, 9, 15), date(2026, 9, 21)),
    ("W39 (22-28 Sep)", date(2026, 9, 22), date(2026, 9, 28)),
    ("W40 (29 Sep-05 Oct)", date(2026, 9, 29), date(2026, 10, 5)),
    ("W41 (06-12 Oct)*", date(2026, 10, 6), date(2026, 10, 12)), # Current reporting week
    ("W42 (13-19 Oct)", date(2026, 10, 13), date(2026, 10, 19)),
    ("W43 (20-26 Oct)", date(2026, 10, 20), date(2026, 10, 26)),
    ("W44 (27 Oct-02 Nov)", date(2026, 10, 27), date(2026, 11, 2)),
    ("W45 (03-09 Nov)", date(2026, 11, 3), date(2026, 11, 9)),
    ("W46 (10-16 Nov)", date(2026, 11, 10), date(2026, 11, 16)),
]

# Raw Tasks Catalog (45 items mapped 100% to source tracker)
TASK_CATALOG = {
    # STREAM 1
    "MR-01": {"task_name": "Prototype – Testing Version 1 using September data", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 1), "end_date": date(2026, 9, 10), "progress": 1.0, "deliverable": "Working ML prototype & benchmarks", "notes": "Prototype built, tested, and validated against baseline September shipment pricing data."},
    "MR-02": {"task_name": "Build and train model on past 6 months of data", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 11), "end_date": date(2026, 9, 20), "progress": 1.0, "deliverable": "Trained ML model weights & pipeline", "notes": "Feature engineering, OrdinalEncoding, and Stage-1A regressor & 1B calibrated elasticity classifier trained on past 6 months."},
    "MR-03": {"task_name": "Internal demo and review with Pricing Team", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 21), "end_date": date(2026, 9, 23), "progress": 1.0, "deliverable": "Stakeholder sign-off & feedback matrix", "notes": "Detailed walkthrough with commercial pricing stakeholders. Feedback gathered on EV curve bounds and market clearing behavior."},
    "MR-04": {"task_name": "Revamp model logic based on feedback", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 24), "end_date": date(2026, 9, 29), "progress": 1.0, "deliverable": "Revamped algorithmic engine logic", "notes": "Refined elasticity curve monotonicity, clearing margin bounds, and balanced EV recommendation logic."},
    "MR-05": {"task_name": "Testing of revamped model", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 30), "end_date": date(2026, 10, 3), "progress": 1.0, "deliverable": "Model validation test suite", "notes": "Edge-case stress testing, historical backtesting, and negative elasticity monotonic verification passed 100%."},
    "MR-06": {"task_name": "UI revamp", "owner": "Deepak", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 10, 4), "end_date": date(2026, 10, 7), "progress": 1.0, "deliverable": "Production-ready Web UI interface", "notes": "Executive UI built adhering to cyan/lime design principles, responsive inputs, and real-time EV curves."},
    "MR-07": {"task_name": "Deployment", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 10, 8), "end_date": date(2026, 10, 10), "progress": 1.0, "deliverable": "Live production pricing engine service", "notes": "Successfully packaged, containerized, and deployed to live production environment. Fully operational."},

    # STREAM 2
    "BP-01": {"task_name": "Frontend – development", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 28), "end_date": date(2026, 10, 4), "progress": 1.0, "deliverable": "Functional frontend booking portal", "notes": "Customer booking interface, cargo specification forms, origin/destination inputs built."},
    "BP-02": {"task_name": "Frontend – changes & UI/UX testing", "owner": "Deepak", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 10, 5), "end_date": date(2026, 10, 6), "progress": 1.0, "deliverable": "Refined frontend integrated with API schema", "notes": "Changes made based on the backend API and the input fields it requires (1 day)."},
    "BP-03": {"task_name": "Backend – development & testing", "owner": "Deepak", "status": "In Progress", "priority": "P1 - Critical", "start_date": date(2026, 10, 6), "end_date": date(2026, 10, 14), "progress": 0.85, "deliverable": "Active rate calculation backend service", "notes": "Done up to showing the rate from the provided rate sheet (3 days). Next step (forward by email or create a job in Shipmnts) is on hold because Shipmnts has not provided the API."},
    "BP-04": {"task_name": "AWS cloud deployment (frontend + backend)", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 10, 5), "end_date": date(2026, 10, 7), "progress": 1.0, "deliverable": "Active AWS cloud hosting environment", "notes": "AWS cloud hosting, networking, domain security, and automated deployment pipelines configured (2 days)."},
    "BP-05": {"task_name": "Deployment testing", "owner": "Deepak", "status": "Planned", "priority": "P2 - High", "start_date": date(2026, 10, 19), "end_date": date(2026, 10, 21), "progress": 0.0, "deliverable": "Deployment testing sign-off", "notes": "Planned 2-day sprint once Shipmnts API endpoints are integrated."},
    "BP-06": {"task_name": "End-to-end application testing", "owner": "Deepak", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 10, 7), "end_date": date(2026, 10, 9), "progress": 1.0, "deliverable": "E2E developer test sign-off", "notes": "Developer testing first (2 days), then testing and review with the internal team."},
    "BP-07": {"task_name": "Approved rate sheet with markup", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 29), "end_date": date(2026, 10, 1), "progress": 1.0, "deliverable": "Approved rate sheet integrated into backend", "notes": "Received."},
    "BP-08": {"task_name": "Client rate sync via Shipmnts API", "owner": "Shipmnts Team", "status": "Waiting on Dependency", "priority": "P1 - Critical", "start_date": date(2026, 10, 11), "end_date": date(2026, 10, 18), "progress": 0.0, "deliverable": "Shipmnts client rate API credentials", "notes": "Read and verify each client's contractual rates. Waiting on the Shipmnts API."},
    "BP-09": {"task_name": "Post-booking workflow via Shipmnts API", "owner": "Shipmnts Team", "status": "Waiting on Dependency", "priority": "P1 - Critical", "start_date": date(2026, 10, 13), "end_date": date(2026, 10, 20), "progress": 0.0, "deliverable": "Shipmnts booking creation API endpoint", "notes": "Pick one approach: (A) trigger an email / create an inquiry, or (B) create the job directly in the system. Blocked: Shipmnts has not provided the API yet."},
    "BP-10": {"task_name": "Show full price breakdown during booking?", "owner": "Deepak / Tyrus", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 9, 25), "end_date": date(2026, 9, 27), "progress": 1.0, "deliverable": "Architectural decision documented", "notes": "Decided."},
    "BP-11": {"task_name": "Show additional / ancillary fees before confirmation?", "owner": "Deepak / Tyrus", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 9, 26), "end_date": date(2026, 9, 28), "progress": 1.0, "deliverable": "Architectural decision documented", "notes": "Decided."},
    "BP-12": {"task_name": "Booking confirmation message", "owner": "Deepak / Tyrus", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 9, 27), "end_date": date(2026, 9, 28), "progress": 1.0, "deliverable": "Confirmation UX copy finalized", "notes": "Decided. Show: 'You will receive an email from our side and your shipment is booked.'"},
    "BP-13": {"task_name": "Inquiry-only flow (no instant booking)", "owner": "Deepak", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 9, 26), "end_date": date(2026, 9, 28), "progress": 1.0, "deliverable": "Heavy cargo inquiry capture workflow", "notes": "After collecting shipment details, tell the user a sales executive will contact them shortly."},

    # STREAM 3
    "CP-01": {"task_name": "Frontend – UI/UX development & testing", "owner": "Deepak", "status": "Planned", "priority": "P2 - High", "start_date": date(2026, 10, 16), "end_date": date(2026, 10, 16), "progress": 0.0, "deliverable": "Courier UI frontend components", "notes": "Rapid 1-day UI build leveraging existing reusable design components."},
    "CP-02": {"task_name": "Backend – development", "owner": "Deepak", "status": "Planned", "priority": "P1 - Critical", "start_date": date(2026, 10, 17), "end_date": date(2026, 10, 17), "progress": 0.0, "deliverable": "Courier pricing calculation service", "notes": "Rule-based courier volumetric weight & tier calculation engine (1 day)."},
    "CP-03": {"task_name": "Deployment", "owner": "Deepak", "status": "Planned", "priority": "P2 - High", "start_date": date(2026, 10, 18), "end_date": date(2026, 10, 18), "progress": 0.0, "deliverable": "Deployed courier engine endpoints", "notes": "CI/CD deployment to existing cloud container infrastructure (1 day)."},
    "CP-04": {"task_name": "End-to-end testing", "owner": "Deepak", "status": "Planned", "priority": "P2 - High", "start_date": date(2026, 10, 19), "end_date": date(2026, 10, 19), "progress": 0.0, "deliverable": "E2E verification sign-off", "notes": "Verification against commercial tariff benchmarks (1 day)."},
    "CP-05": {"task_name": "Buffer day", "owner": "Deepak", "status": "Planned", "priority": "P3 - Medium", "start_date": date(2026, 10, 20), "end_date": date(2026, 10, 20), "progress": 0.0, "deliverable": "Contingency buffer", "notes": "Contingency day for fixes or delays (1 day)."},
    "CP-06": {"task_name": "Courier rate sheet with markup", "owner": "Commercial Team", "status": "Waiting on Dependency", "priority": "P1 - Critical", "start_date": date(2026, 10, 11), "end_date": date(2026, 10, 15), "progress": 0.0, "deliverable": "Commercial courier markup rate matrix", "notes": "Rate sheet with the markup added. Needed before development starts."},

    # STREAM 4
    "SOP-01": {"task_name": "Billing (Chaitanya)", "owner": "Deepak / Chaitanya", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 9, 20), "end_date": date(2026, 9, 26), "progress": 1.0, "deliverable": "Approved Billing SOP Architecture Diagram", "notes": "End-to-end billing workflow mapped, including approval triggers, dispute handling, and ERP handoff."},
    "SOP-02": {"task_name": "Trade Receivables & Confirmation (Siddhesh)", "owner": "Deepak / Siddhesh", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 9, 27), "end_date": date(2026, 10, 3), "progress": 1.0, "deliverable": "Approved Receivables SOP Diagram", "notes": "Receivables reconciliation, customer confirmation cycles, and audit checkpoints documented."},
    "SOP-03": {"task_name": "Pricing (Narendra)", "owner": "Deepak / Narendra", "status": "Planned", "priority": "P2 - High", "start_date": date(2026, 10, 16), "end_date": date(2026, 10, 22), "progress": 0.0, "deliverable": "Pricing Operations SOP Document", "notes": "Scheduled working session with Narendra to map pricing approvals, rate refreshes, and margin rules."},

    # STREAM 5
    "AI-01": {"task_name": "Build in-house AI tool (working alongside Priyank)", "owner": "Deepak / Priyank", "status": "Planned", "priority": "P2 - High", "start_date": date(2026, 10, 20), "end_date": date(2026, 11, 6), "progress": 0.0, "deliverable": "Proprietary AI email-to-quote architecture", "notes": "Plan to build once we finalize one external tool, so we can compare and build on that basis."},

    # STREAM 6
    "VE-01": {"task_name": "Nexcade.ai", "owner": "Deepak", "status": "In Progress", "priority": "P2 - High", "start_date": date(2026, 10, 4), "end_date": date(2026, 10, 14), "progress": 0.70, "deliverable": "Nexcade.ai technical evaluation report", "notes": "First call done; second call scheduled for 08 Oct 2026. Evaluating AI parsing accuracy."},
    "VE-02": {"task_name": "cargo.one", "owner": "Deepak", "status": "Waiting on Dependency", "priority": "P3 - Medium", "start_date": date(2026, 10, 5), "end_date": date(2026, 10, 16), "progress": 0.10, "deliverable": "cargo.one capability scorecard", "notes": "Demo requested; no response yet. Deepak following up."},
    "VE-03": {"task_name": "CargoMart", "owner": "Deepak", "status": "In Progress", "priority": "P2 - High", "start_date": date(2026, 10, 3), "end_date": date(2026, 10, 13), "progress": 0.75, "deliverable": "CargoMart assessment matrix", "notes": "Demo booked and attended on 07 Oct 2026. Reviewing API capabilities and rate coverage."},
    "VE-04": {"task_name": "RAFT", "owner": "Deepak", "status": "Waiting on Dependency", "priority": "P3 - Medium", "start_date": date(2026, 10, 6), "end_date": date(2026, 10, 17), "progress": 0.10, "deliverable": "RAFT feature matrix", "notes": "Explored the platform; demo requested, response not yet received."},

    # STREAM 7
    "EN-01": {"task_name": "Shipmnts tutorial", "owner": "Deepak", "status": "Planned", "priority": "P3 - Medium", "start_date": date(2026, 10, 23), "end_date": date(2026, 10, 29), "progress": 0.0, "deliverable": "Operational user guide & walkthrough", "notes": "Needs to be started. Step-by-step training deck and walkthrough for operations and pricing teams."},

    # STREAM 8
    "EM-01": {"task_name": "Zoho Infra & CRM", "owner": "Deepak / Tyra", "status": "Needs Decision", "priority": "P1 - Critical", "start_date": date(2026, 9, 28), "end_date": date(2026, 10, 14), "progress": 0.90, "deliverable": "Zoho comparative scorecard", "notes": "Demo done. Comparing pricing with MailerCloud to proceed with one."},
    "EM-02": {"task_name": "MailerCloud", "owner": "Deepak / Tyra", "status": "Needs Decision", "priority": "P1 - Critical", "start_date": date(2026, 9, 28), "end_date": date(2026, 10, 14), "progress": 0.90, "deliverable": "MailerCloud comparative scorecard", "notes": "Demo done. Comparing pricing with Zoho to proceed with one."},

    # STREAM 9
    "LD-01": {"task_name": "Get external lead database (Apollo DB)", "owner": "Tyra / External Data Team", "status": "Waiting on Dependency", "priority": "P1 - Critical", "start_date": date(2026, 10, 5), "end_date": date(2026, 10, 14), "progress": 0.0, "deliverable": "Apollo DB raw export file", "notes": "Waiting to receive it. Essential input for multi-source data merge."},
    "LD-02": {"task_name": "Get shipmnts data for existing customers", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 22), "end_date": date(2026, 9, 26), "progress": 1.0, "deliverable": "Clean existing-customer shipping dataset", "notes": "Internal shipping histories extracted, schema structured, and verified for lead mining."},
    "LD-03": {"task_name": "Data cleaning, structuring & merging (Shipmnts, External Source, Data Miner)", "owner": "Deepak", "status": "Planned", "priority": "P1 - Critical", "start_date": date(2026, 10, 15), "end_date": date(2026, 10, 15), "progress": 0.0, "deliverable": "Unified cleansed customer lead database", "notes": "Needs the external database first. Together with getting the database, about 1 day."},
    "LD-04": {"task_name": "UI build", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 10, 1), "end_date": date(2026, 10, 6), "progress": 1.0, "deliverable": "Functional Lead Engine Web UI", "notes": "Modern interactive UI built allowing sales reps to search, filter, and inspect customer lead segments."},
    "LD-05": {"task_name": "Backend development", "owner": "Deepak", "status": "In Progress", "priority": "P1 - Critical", "start_date": date(2026, 10, 7), "end_date": date(2026, 10, 14), "progress": 0.80, "deliverable": "Backend lead scoring service", "notes": "Done up to adding the rate sheet; the rate sheet has now been received."},
    "LD-06": {"task_name": "Testing", "owner": "Deepak", "status": "Completed", "priority": "P2 - High", "start_date": date(2026, 10, 7), "end_date": date(2026, 10, 9), "progress": 1.0, "deliverable": "Backend rate calculation sign-off", "notes": "Tested backend calculation logic and verified rate display accuracy."},
    "LD-07": {"task_name": "Approved rate sheet with markup", "owner": "Deepak", "status": "Completed", "priority": "P1 - Critical", "start_date": date(2026, 9, 27), "end_date": date(2026, 9, 30), "progress": 1.0, "deliverable": "Approved markup rate sheet", "notes": "Received."},
    "LD-08": {"task_name": "Deployment", "owner": "Deepak", "status": "Planned", "priority": "P1 - Critical", "start_date": date(2026, 10, 16), "end_date": date(2026, 10, 16), "progress": 0.0, "deliverable": "Live production lead engine platform", "notes": "Rapid 1-day deployment scheduled immediately following database merge."}
}

# 9 Project Groups with Sections & Tasks
PROJECT_GROUPS = [
    {
        "id": "PRJ-01",
        "name": "Project 1: Margin Recommender (Pricing Engine)",
        "portfolio": "Tyrus",
        "owner": "Deepak",
        "status": "Completed",
        "priority": "P1 - Critical",
        "sections": [
            ("Core Machine Learning Pipeline & Production Release", ["MR-01", "MR-02", "MR-03", "MR-04", "MR-05", "MR-06", "MR-07"])
        ]
    },
    {
        "id": "PRJ-02",
        "name": "Project 2: Booking Platform (Website)",
        "portfolio": "Tyrus",
        "owner": "Deepak",
        "status": "In Progress",
        "priority": "P1 - Critical",
        "sections": [
            ("Build & Delivery Plan (about 10 days of effort)", ["BP-01", "BP-02", "BP-03", "BP-04", "BP-05", "BP-06"]),
            ("Air Cargo under 500 kg – Dependencies", ["BP-07", "BP-08", "BP-09"]),
            ("Air Cargo under 500 kg – Key Decisions", ["BP-10", "BP-11", "BP-12"]),
            ("Air Cargo above 500 kg – Inquiry Flow", ["BP-13"])
        ]
    },
    {
        "id": "PRJ-03",
        "name": "Project 3: Courier Pricing Engine",
        "portfolio": "Tyrus",
        "owner": "Deepak",
        "status": "Planned",
        "priority": "P1 - Critical",
        "sections": [
            ("Prerequisite (Needed before development starts)", ["CP-06"]),
            ("Build & Delivery Plan (about 5 days of effort)", ["CP-01", "CP-02", "CP-03", "CP-04", "CP-05"])
        ]
    },
    {
        "id": "PRJ-04",
        "name": "Project 4: SOP Flow Charts (Operational Governance)",
        "portfolio": "Tyrus",
        "owner": "Deepak / Cross-functional",
        "status": "Completed",
        "priority": "P2 - High",
        "sections": [
            ("Operational Standardization & SOP Mapping", ["SOP-01", "SOP-02", "SOP-03"])
        ]
    },
    {
        "id": "PRJ-05",
        "name": "Project 5: Inhouse AI Tool (Inquiry & Email Intelligence)",
        "portfolio": "Tyrus",
        "owner": "Deepak / Priyank",
        "status": "Planned",
        "priority": "P2 - High",
        "sections": [
            ("Strategic R&D & Benchmarking", ["AI-01"])
        ]
    },
    {
        "id": "PRJ-06",
        "name": "Project 6: Tool Suggestion (Vendor Evaluation)",
        "portfolio": "Tyrus",
        "owner": "Deepak",
        "status": "In Progress",
        "priority": "P2 - High",
        "sections": [
            ("Market Tech Intelligence & Vendor Scoring", ["VE-01", "VE-02", "VE-03", "VE-04"])
        ]
    },
    {
        "id": "PRJ-07",
        "name": "Project 7: Systems Training & Operations Enablement",
        "portfolio": "Tyrus",
        "owner": "Deepak",
        "status": "Planned",
        "priority": "P3 - Medium",
        "sections": [
            ("Team Enablement & Training Handbook", ["EN-01"])
        ]
    },
    {
        "id": "PRJ-08",
        "name": "Project 8: Email Infrastructure Architecture",
        "portfolio": "Tyra",
        "owner": "Deepak / Tyra",
        "status": "Needs Decision",
        "priority": "P1 - Critical",
        "sections": [
            ("Outbound Messaging Platform Selection", ["EM-01", "EM-02"])
        ]
    },
    {
        "id": "PRJ-09",
        "name": "Project 9: Lead Generation & Customer Intelligence Engine",
        "portfolio": "Tyra",
        "owner": "Deepak",
        "status": "In Progress",
        "priority": "P1 - Critical",
        "sections": [
            ("Customer Lead Data Pipeline & Application Build", ["LD-01", "LD-02", "LD-03", "LD-04", "LD-05", "LD-06", "LD-07", "LD-08"])
        ]
    }
]

def style_cell(cell, font=None, fill=None, alignment=None, border=None):
    if font: cell.font = font
    if fill: cell.fill = fill
    if alignment: cell.alignment = alignment
    if border: cell.border = border

wb = openpyxl.Workbook()
wb.remove(wb.active)

# ==============================================================================
# TAB 1: GROUPED GANTT CHART
# ==============================================================================
print("Generating Sheet 1: Grouped Gantt Chart...")
ws_gantt = wb.create_sheet(title="Gantt Chart")
ws_gantt.views.sheetView[0].showGridLines = True
ws_gantt.sheet_properties.outlinePr.summaryBelow = False

# Title Banner
ws_gantt.merge_cells("B2:W2")
g_title = ws_gantt["B2"]
g_title.value = "DEEPAK  |  EXECUTIVE GANTT CHART & PROJECT WORK STATUS TRACKER"
style_cell(g_title, font=Font(name=FONT_FAMILY, size=15, bold=True, color=COLOR_WHITE),
           fill=PatternFill("solid", fgColor=COLOR_NAVY),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_gantt.row_dimensions[2].height = 36

ws_gantt.merge_cells("B3:W3")
g_sub = ws_gantt["B3"]
g_sub.value = "GROUPED BY STRATEGIC PROJECT & WORK CATEGORY  |  AS OF 11 OCTOBER 2026  |  CLICK [-]/[+] ON LEFT MARGIN TO EXPAND OR COLLAPSE"
style_cell(g_sub, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN),
           fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_gantt.row_dimensions[3].height = 20

# Top KPI Metric Ribbon (Row 5 to 7)
kpi_ribbon = [
    ("B5:C5", "B6:C6", "B7:C7", "TOTAL PROJECTS", 9, "Strategic Workstreams", COLOR_CYAN, "0284C7"),
    ("D5:F5", "D6:F6", "D7:F7", "TOTAL DELIVERABLES", 45, "All Tracked Work Items", COLOR_CYAN_DARK, "0369A1"),
    ("G5:I5", "G6:I6", "G7:I7", "COMPLETED MILESTONES", "22 (48.9%)", "100% Signed Off & Live", COLOR_LIME, "16A34A"),
    ("J5:L5", "J6:L6", "J7:L7", "ACTIVE IN PROGRESS", "4 (8.9%)", "Active Development / Discovery", "F59E0B", "D97706"),
    ("M5:P5", "M6:P6", "M7:P7", "EXTERNAL DEPENDENCIES", "6 (13.3%)", "Blocked on Shipmnts API / Data", "EF4444", "DC2626"),
    ("Q5:S5", "Q6:S6", "Q7:S7", "UNDER DECISION", "2 (4.4%)", "Zoho vs MailerCloud", "8B5CF6", "7C3AED"),
    ("T5:W5", "T6:W6", "T7:W7", "PLANNED / SPRINT READY", "11 (24.4%)", "5-Day Courier & 1-Day Lead", "6366F1", "4F46E5"),
]

for label_range, val_range, sub_range, label, val_text, sub_text, top_col, text_col in kpi_ribbon:
    ws_gantt.merge_cells(label_range)
    c_lbl = ws_gantt[label_range.split(":")[0]]
    c_lbl.value = label
    style_cell(c_lbl, font=Font(name=FONT_FAMILY, size=8, bold=True, color=COLOR_MUTED_TEXT),
               fill=PatternFill("solid", fgColor=COLOR_BG_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=top_col), left=Side(style='thin', color=COLOR_BORDER_LIGHT), right=Side(style='thin', color=COLOR_BORDER_LIGHT)))

    ws_gantt.merge_cells(val_range)
    c_val = ws_gantt[val_range.split(":")[0]]
    c_val.value = val_text
    style_cell(c_val, font=Font(name=FONT_FAMILY, size=16, bold=True, color=text_col),
               fill=PatternFill("solid", fgColor=COLOR_BG_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(left=Side(style='thin', color=COLOR_BORDER_LIGHT), right=Side(style='thin', color=COLOR_BORDER_LIGHT)))

    ws_gantt.merge_cells(sub_range)
    c_sub = ws_gantt[sub_range.split(":")[0]]
    c_sub.value = sub_text
    style_cell(c_sub, font=Font(name=FONT_FAMILY, size=8, color=COLOR_MUTED_TEXT),
               fill=PatternFill("solid", fgColor=COLOR_BG_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(left=Side(style='thin', color=COLOR_BORDER_LIGHT), right=Side(style='thin', color=COLOR_BORDER_LIGHT), bottom=Side(style='thin', color=COLOR_BORDER_LIGHT)))

ws_gantt.row_dimensions[5].height = 18
ws_gantt.row_dimensions[6].height = 26
ws_gantt.row_dimensions[7].height = 18

# Gantt Table Headers (Row 9)
gantt_headers = [
    ("B9", "WBS / ID", 11, "center"),
    ("C9", "Strategic Project & Work Item", 42, "left"),
    ("D9", "Portfolio", 10, "center"),
    ("E9", "Owner / Lead", 14, "left"),
    ("F9", "Status", 18, "center"),
    ("G9", "Priority", 12, "center"),
    ("H9", "Start Date", 12, "center"),
    ("I9", "End Date", 12, "center"),
    ("J9", "Duration", 10, "center"),
    ("K9", "Progress", 10, "center"),
]

for idx, (w_label, w_start, w_end) in enumerate(GANTT_WEEKS):
    col_idx = 12 + idx # Col L is 12
    col_letter = get_column_letter(col_idx)
    gantt_headers.append((f"{col_letter}9", w_label, 12, "center"))

for pos, h_title, col_w, align in gantt_headers:
    cell = ws_gantt[pos]
    cell.value = h_title
    is_curr_week = "*" in h_title
    bg_color = COLOR_CYAN_DARK if is_curr_week else COLOR_NAVY
    text_color = COLOR_WHITE
    style_cell(cell, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=text_color),
               fill=PatternFill("solid", fgColor=bg_color),
               alignment=Alignment(horizontal=align, vertical="center", wrap_text=True),
               border=border_header)
    col_let = pos[0] if len(pos) == 2 else pos[:2]
    ws_gantt.column_dimensions[col_let].width = col_w

ws_gantt.row_dimensions[9].height = 28

# Populate Grouped Projects and Tasks
current_row = 10
project_chart_data = [] # To store project summary info for native BarChart

for p_group in PROJECT_GROUPS:
    prj_id = p_group["id"]
    prj_name = p_group["name"]
    portfolio = p_group["portfolio"]
    owner = p_group["owner"]
    prj_status = p_group["status"]
    prj_prio = p_group["priority"]
    sections = p_group["sections"]

    # Gather all tasks for this project
    prj_task_ids = [tid for _, t_ids in sections for tid in t_ids]
    prj_tasks = [TASK_CATALOG[tid] for tid in prj_task_ids]

    earliest_start = min(t["start_date"] for t in prj_tasks)
    latest_end = max(t["end_date"] for t in prj_tasks)
    total_span_days = (latest_end - earliest_start).days + 1
    avg_progress = sum(t["progress"] for t in prj_tasks) / len(prj_tasks)

    # Save for BarChart
    project_chart_data.append((prj_name, earliest_start, latest_end, total_span_days))

    # --- PROJECT HEADER ROW ---
    prj_row = current_row
    ws_gantt.row_dimensions[prj_row].height = 24
    ws_gantt.row_dimensions[prj_row].outline_level = 0

    st_cfg = STATUS_STYLES.get(prj_status, STATUS_STYLES["Planned"])

    # B: ID
    ws_gantt[f"B{prj_row}"] = prj_id
    style_cell(ws_gantt[f"B{prj_row}"], font=Font(name=FONT_FAMILY, size=9, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # C: Project Name
    ws_gantt[f"C{prj_row}"] = prj_name.upper()
    style_cell(ws_gantt[f"C{prj_row}"], font=Font(name=FONT_FAMILY, size=9.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="left", vertical="center", indent=1),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # D: Portfolio
    ws_gantt[f"D{prj_row}"] = portfolio
    style_cell(ws_gantt[f"D{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # E: Owner
    ws_gantt[f"E{prj_row}"] = owner
    style_cell(ws_gantt[f"E{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="left", vertical="center", indent=1),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # F: Status Badge
    ws_gantt[f"F{prj_row}"] = prj_status
    style_cell(ws_gantt[f"F{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=st_cfg["font"]),
               fill=PatternFill("solid", fgColor=st_cfg["fill"]),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # G: Priority
    ws_gantt[f"G{prj_row}"] = prj_prio
    style_cell(ws_gantt[f"G{prj_row}"], font=Font(name=FONT_FAMILY, size=8, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # H: Earliest Start Date
    ws_gantt[f"H{prj_row}"] = earliest_start
    style_cell(ws_gantt[f"H{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))
    ws_gantt[f"H{prj_row}"].number_format = 'yyyy-mm-dd'

    # I: Latest End Date
    ws_gantt[f"I{prj_row}"] = latest_end
    style_cell(ws_gantt[f"I{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))
    ws_gantt[f"I{prj_row}"].number_format = 'yyyy-mm-dd'

    # J: Total Span Duration Formula
    ws_gantt[f"J{prj_row}"] = f'=(I{prj_row}-H{prj_row})+1'
    style_cell(ws_gantt[f"J{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    # K: Overall Progress %
    ws_gantt[f"K{prj_row}"] = avg_progress
    style_cell(ws_gantt[f"K{prj_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal="center", vertical="center"),
               border=Border(top=Side(style='medium', color=COLOR_CYAN), left=Side(style='thin', color=COLOR_NAVY_LIGHT), right=Side(style='thin', color=COLOR_NAVY_LIGHT), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))
    ws_gantt[f"K{prj_row}"].number_format = '0%'

    # L to V: Project Span Bar
    for w_idx, (w_label, w_start, w_end) in enumerate(GANTT_WEEKS):
        col_idx = 12 + w_idx
        col_letter = get_column_letter(col_idx)
        cell = ws_gantt[f"{col_letter}{prj_row}"]
        overlap = max(earliest_start, w_start) <= min(latest_end, w_end)

        if overlap:
            cell.value = "████"
            style_cell(cell, font=Font(name=FONT_FAMILY, size=9, bold=True, color=COLOR_CYAN),
                       fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
                       alignment=Alignment(horizontal="center", vertical="center"),
                       border=Border(top=Side(style='medium', color=COLOR_CYAN), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))
        else:
            style_cell(cell, fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
                       border=Border(top=Side(style='medium', color=COLOR_CYAN), bottom=Side(style='thin', color=COLOR_NAVY_LIGHT)))

    current_row += 1

    # --- SECTIONS & SUB-TASKS ---
    for sec_name, t_ids in sections:
        if sec_name:
            # Section Header Row
            sec_row = current_row
            ws_gantt.row_dimensions[sec_row].height = 20
            ws_gantt.row_dimensions[sec_row].outline_level = 1

            ws_gantt.merge_cells(f"C{sec_row}:K{sec_row}")
            ws_gantt[f"B{sec_row}"] = "▸"
            style_cell(ws_gantt[f"B{sec_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN_DARK),
                       fill=PatternFill("solid", fgColor=COLOR_BG_MUTED),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

            ws_gantt[f"C{sec_row}"] = f"{sec_name}"
            style_cell(ws_gantt[f"C{sec_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, italic=True, color="334155"),
                       fill=PatternFill("solid", fgColor=COLOR_BG_MUTED),
                       alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

            for w_idx in range(len(GANTT_WEEKS)):
                col_idx = 12 + w_idx
                col_letter = get_column_letter(col_idx)
                c_sec_w = ws_gantt[f"{col_letter}{sec_row}"]
                style_cell(c_sec_w, fill=PatternFill("solid", fgColor=COLOR_BG_MUTED), border=border_thin)

            current_row += 1

        # Child Tasks
        for tid in t_ids:
            task = TASK_CATALOG[tid]
            t_row = current_row
            ws_gantt.row_dimensions[t_row].height = 20
            ws_gantt.row_dimensions[t_row].outline_level = 1

            t_status = task["status"]
            t_st_cfg = STATUS_STYLES.get(t_status, STATUS_STYLES["Planned"])
            t_start = task["start_date"]
            t_end = task["end_date"]
            t_prog = task["progress"]

            # B: Task ID
            ws_gantt[f"B{t_row}"] = tid
            style_cell(ws_gantt[f"B{t_row}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_NAVY),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

            # C: Indented Task Name
            ws_gantt[f"C{t_row}"] = f"    •  {task['task_name']}"
            style_cell(ws_gantt[f"C{t_row}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_DARK_TEXT),
                       alignment=Alignment(horizontal="left", vertical="center"), border=border_thin)

            # D: Portfolio
            ws_gantt[f"D{t_row}"] = portfolio
            style_cell(ws_gantt[f"D{t_row}"], font=Font(name=FONT_FAMILY, size=8),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

            # E: Owner
            ws_gantt[f"E{t_row}"] = task["owner"]
            style_cell(ws_gantt[f"E{t_row}"], font=Font(name=FONT_FAMILY, size=8),
                       alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

            # F: Status Badge
            ws_gantt[f"F{t_row}"] = t_status
            style_cell(ws_gantt[f"F{t_row}"], font=Font(name=FONT_FAMILY, size=8, bold=True, color=t_st_cfg["font"]),
                       fill=PatternFill("solid", fgColor=t_st_cfg["fill"]),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

            # G: Priority
            ws_gantt[f"G{t_row}"] = task["priority"]
            style_cell(ws_gantt[f"G{t_row}"], font=Font(name=FONT_FAMILY, size=8),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

            # H: Start Date
            ws_gantt[f"H{t_row}"] = t_start
            style_cell(ws_gantt[f"H{t_row}"], font=Font(name=FONT_FAMILY, size=8),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)
            ws_gantt[f"H{t_row}"].number_format = 'yyyy-mm-dd'

            # I: End Date
            ws_gantt[f"I{t_row}"] = t_end
            style_cell(ws_gantt[f"I{t_row}"], font=Font(name=FONT_FAMILY, size=8),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)
            ws_gantt[f"I{t_row}"].number_format = 'yyyy-mm-dd'

            # J: Duration Formula
            ws_gantt[f"J{t_row}"] = f'=(I{t_row}-H{t_row})+1'
            style_cell(ws_gantt[f"J{t_row}"], font=Font(name=FONT_FAMILY, size=8, bold=True),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

            # K: Progress
            ws_gantt[f"K{t_row}"] = t_prog
            style_cell(ws_gantt[f"K{t_row}"], font=Font(name=FONT_FAMILY, size=8, bold=True),
                       alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)
            ws_gantt[f"K{t_row}"].number_format = '0%'

            # L to V: Visual Gantt Week Blocks
            for w_idx, (w_label, w_start, w_end) in enumerate(GANTT_WEEKS):
                col_idx = 12 + w_idx
                col_letter = get_column_letter(col_idx)
                cell = ws_gantt[f"{col_letter}{t_row}"]
                overlap = max(t_start, w_start) <= min(t_end, w_end)
                is_curr = "*" in w_label

                if overlap:
                    cell.value = "████"
                    style_cell(cell, font=Font(name=FONT_FAMILY, size=9, bold=True, color=t_st_cfg["bar_fill"]),
                               fill=PatternFill("solid", fgColor=t_st_cfg["fill"]),
                               alignment=Alignment(horizontal="center", vertical="center"),
                               border=Border(left=Side(style='thin', color=t_st_cfg["bar_fill"]),
                                             right=Side(style='thin', color=t_st_cfg["bar_fill"]),
                                             top=Side(style='thin', color=COLOR_BORDER_LIGHT),
                                             bottom=Side(style='thin', color=COLOR_BORDER_LIGHT)))
                else:
                    bg = "FFFFFF" if not is_curr else "F8FAFC"
                    style_cell(cell, fill=PatternFill("solid", fgColor=bg), border=border_thin)

            current_row += 1

# Native BarChart data summary table in cols Y to AA (Rows 10 to 18)
ws_gantt["Y9"] = "Project Summary"
ws_gantt["Z9"] = "Start Offset (Days)"
ws_gantt["AA9"] = "Duration (Days)"
for c_hdr in ["Y9", "Z9", "AA9"]:
    style_cell(ws_gantt[c_hdr], font=Font(name=FONT_FAMILY, size=8, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY), alignment=Alignment(horizontal="center", vertical="center"))

base_date = date(2026, 9, 1)
for s_idx, (p_name, p_st, p_en, p_dur) in enumerate(project_chart_data, start=10):
    ws_gantt[f"Y{s_idx}"] = p_name
    ws_gantt[f"Z{s_idx}"] = (p_st - base_date).days
    ws_gantt[f"AA{s_idx}"] = p_dur
    for col_l in ["Y", "Z", "AA"]:
        style_cell(ws_gantt[f"{col_l}{s_idx}"], font=Font(name=FONT_FAMILY, size=8), border=border_thin)

# Create Native Stacked Bar Chart (Gantt Chart)
chart = BarChart()
chart.type = "bar"
chart.grouping = "stacked"
chart.overlap = 100
chart.title = "High-Level Strategic Project Gantt Timeline (Q3–Q4 2026)"
chart.style = 10
chart.width = 24
chart.height = 12

chart_data = Reference(ws_gantt, min_col=26, min_row=9, max_col=27, max_row=18)
chart_cats = Reference(ws_gantt, min_col=25, min_row=10, max_row=18)
chart.add_data(chart_data, titles_from_data=True)
chart.set_categories(chart_cats)

if len(chart.series) >= 2:
    s_offset = chart.series[0]
    s_offset.graphicalProperties.noFill = True
    s_offset.graphicalProperties.line.noFill = True

    s_dur = chart.series[1]
    s_dur.graphicalProperties.solidFill = COLOR_CYAN
    s_dur.graphicalProperties.line.solidFill = COLOR_CYAN_DARK

chart.x_axis.title = "Days from 01 Sep 2026"
chart.legend = None

# Anchor chart below all rows (current_row + 2)
chart_anchor_row = current_row + 2
ws_gantt.add_chart(chart, f"B{chart_anchor_row}")


# ==============================================================================
# TAB 2: WORK WITH TYRUS (DETAILED BREAKDOWN - 35 TASKS)
# ==============================================================================
print("Generating Sheet 2: Work with Tyrus (Detail)...")
ws_tyrus = wb.create_sheet(title="Work with Tyrus")
ws_tyrus.views.sheetView[0].showGridLines = True

ws_tyrus.merge_cells("B2:J2")
t_hdr = ws_tyrus["B2"]
t_hdr.value = "WORK WITH TYRUS  |  PROJECT STATUS & DELIVERABLE MASTER TRACKER"
style_cell(t_hdr, font=Font(name=FONT_FAMILY, size=15, bold=True, color=COLOR_WHITE),
           fill=PatternFill("solid", fgColor=COLOR_NAVY),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_tyrus.row_dimensions[2].height = 36

ws_tyrus.merge_cells("B3:J3")
t_sub = ws_tyrus["B3"]
t_sub.value = "35 TASKS ACROSS 7 STRATEGIC MODULES: PRICING ENGINE, BOOKING PLATFORM, COURIER, SOPs, AI TOOL, VENDORS & TRAINING"
style_cell(t_sub, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN),
           fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_tyrus.row_dimensions[3].height = 20

tyrus_headers = [
    ("B4", "#", 5, "center"),
    ("C4", "Workstream / Module", 28, "left"),
    ("D4", "Category / Phase", 24, "left"),
    ("E4", "Deliverable / Task Description", 38, "left"),
    ("F4", "Status", 18, "center"),
    ("G4", "Timeline / Days", 15, "center"),
    ("H4", "Priority", 12, "center"),
    ("I4", "Key Output / Milestone", 32, "left"),
    ("J4", "Detailed Progress Notes & Actionable Next Steps", 55, "left"),
]

for pos, h_title, col_w, align in tyrus_headers:
    cell = ws_tyrus[pos]
    cell.value = h_title
    style_cell(cell, font=Font(name=FONT_FAMILY, size=9, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal=align, vertical="center"),
               border=border_header)
    col_let = pos[0] if len(pos) == 2 else pos[:2]
    ws_tyrus.column_dimensions[col_let].width = col_w

ws_tyrus.row_dimensions[4].height = 26

# Filter Tyrus tasks (35 tasks)
tyrus_task_ids = [tid for p in PROJECT_GROUPS if p["portfolio"] == "Tyrus" for _, t_ids in p["sections"] for tid in t_ids]
assert len(tyrus_task_ids) == 35, f"Expected 35 Tyrus tasks, got {len(tyrus_task_ids)}"

for idx, tid in enumerate(tyrus_task_ids, start=5):
    task = TASK_CATALOG[tid]
    # find stream name and sub-category
    stream_name = next(p["name"].replace("Project ", "") for p in PROJECT_GROUPS if tid in [t for _, ts in p["sections"] for t in ts])
    sub_cat = next(sec for p in PROJECT_GROUPS for sec, ts in p["sections"] if tid in ts)

    st_cfg = STATUS_STYLES.get(task["status"], STATUS_STYLES["Planned"])

    ws_tyrus[f"B{idx}"] = idx - 4
    style_cell(ws_tyrus[f"B{idx}"], font=Font(name=FONT_FAMILY, size=9, bold=True), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_tyrus[f"C{idx}"] = stream_name
    style_cell(ws_tyrus[f"C{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_MUTED_TEXT), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyrus[f"D{idx}"] = sub_cat
    style_cell(ws_tyrus[f"D{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_CYAN_DARK), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyrus[f"E{idx}"] = task["task_name"]
    style_cell(ws_tyrus[f"E{idx}"], font=Font(name=FONT_FAMILY, size=9, bold=True, color=COLOR_DARK_TEXT), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyrus[f"F{idx}"] = task["status"]
    style_cell(ws_tyrus[f"F{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=st_cfg["font"]),
               fill=PatternFill("solid", fgColor=st_cfg["fill"]),
               alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    dur = (task["end_date"] - task["start_date"]).days + 1
    ws_tyrus[f"G{idx}"] = f"{dur} days" if dur > 1 else "1 day"
    style_cell(ws_tyrus[f"G{idx}"], font=Font(name=FONT_FAMILY, size=8.5), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_tyrus[f"H{idx}"] = task["priority"]
    style_cell(ws_tyrus[f"H{idx}"], font=Font(name=FONT_FAMILY, size=8), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_tyrus[f"I{idx}"] = task["deliverable"]
    style_cell(ws_tyrus[f"I{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_NAVY), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyrus[f"J{idx}"] = task["notes"]
    style_cell(ws_tyrus[f"J{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_MUTED_TEXT), alignment=Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1), border=border_thin)

    ws_tyrus.row_dimensions[idx].height = 24


# ==============================================================================
# TAB 3: WORK WITH TYRA (DETAILED BREAKDOWN - 10 TASKS)
# ==============================================================================
print("Generating Sheet 3: Work with Tyra (Detail)...")
ws_tyra = wb.create_sheet(title="Work with Tyra")
ws_tyra.views.sheetView[0].showGridLines = True

ws_tyra.merge_cells("B2:J2")
ty_hdr = ws_tyra["B2"]
ty_hdr.value = "WORK WITH TYRA  |  EMAIL INFRASTRUCTURE & LEAD INTELLIGENCE ENGINE"
style_cell(ty_hdr, font=Font(name=FONT_FAMILY, size=15, bold=True, color=COLOR_WHITE),
           fill=PatternFill("solid", fgColor=COLOR_NAVY),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_tyra.row_dimensions[2].height = 36

ws_tyra.merge_cells("B3:J3")
ty_sub = ws_tyra["B3"]
ty_sub.value = "10 TASKS COVERING OUTBOUND INFRASTRUCTURE SELECTION & MULTI-SOURCE CUSTOMER LEAD DATA PIPELINE"
style_cell(ty_sub, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN),
           fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_tyra.row_dimensions[3].height = 20

for pos, h_title, col_w, align in tyrus_headers:
    cell = ws_tyra[pos]
    cell.value = h_title
    style_cell(cell, font=Font(name=FONT_FAMILY, size=9, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal=align, vertical="center"),
               border=border_header)
    col_let = pos[0] if len(pos) == 2 else pos[:2]
    ws_tyra.column_dimensions[col_let].width = col_w

ws_tyra.row_dimensions[4].height = 26

# Filter Tyra tasks (10 tasks)
tyra_task_ids = [tid for p in PROJECT_GROUPS if p["portfolio"] == "Tyra" for _, t_ids in p["sections"] for tid in t_ids]
assert len(tyra_task_ids) == 10, f"Expected 10 Tyra tasks, got {len(tyra_task_ids)}"

for idx, tid in enumerate(tyra_task_ids, start=5):
    task = TASK_CATALOG[tid]
    stream_name = next(p["name"].replace("Project ", "") for p in PROJECT_GROUPS if tid in [t for _, ts in p["sections"] for t in ts])
    sub_cat = next(sec for p in PROJECT_GROUPS for sec, ts in p["sections"] if tid in ts)

    st_cfg = STATUS_STYLES.get(task["status"], STATUS_STYLES["Planned"])

    ws_tyra[f"B{idx}"] = idx - 4
    style_cell(ws_tyra[f"B{idx}"], font=Font(name=FONT_FAMILY, size=9, bold=True), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_tyra[f"C{idx}"] = stream_name
    style_cell(ws_tyra[f"C{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_MUTED_TEXT), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyra[f"D{idx}"] = sub_cat
    style_cell(ws_tyra[f"D{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_CYAN_DARK), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyra[f"E{idx}"] = task["task_name"]
    style_cell(ws_tyra[f"E{idx}"], font=Font(name=FONT_FAMILY, size=9, bold=True, color=COLOR_DARK_TEXT), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyra[f"F{idx}"] = task["status"]
    style_cell(ws_tyra[f"F{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=st_cfg["font"]),
               fill=PatternFill("solid", fgColor=st_cfg["fill"]),
               alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    dur = (task["end_date"] - task["start_date"]).days + 1
    ws_tyra[f"G{idx}"] = f"{dur} days" if dur > 1 else "1 day"
    style_cell(ws_tyra[f"G{idx}"], font=Font(name=FONT_FAMILY, size=8.5), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_tyra[f"H{idx}"] = task["priority"]
    style_cell(ws_tyra[f"H{idx}"], font=Font(name=FONT_FAMILY, size=8), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_tyra[f"I{idx}"] = task["deliverable"]
    style_cell(ws_tyra[f"I{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_NAVY), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_tyra[f"J{idx}"] = task["notes"]
    style_cell(ws_tyra[f"J{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_MUTED_TEXT), alignment=Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1), border=border_thin)

    ws_tyra.row_dimensions[idx].height = 24


# ==============================================================================
# TAB 4: DEPENDENCIES & VENDOR SCORECARD
# ==============================================================================
print("Generating Sheet 4: Dependencies & Vendor Matrix...")
ws_dep = wb.create_sheet(title="Dependencies & Vendors")
ws_dep.views.sheetView[0].showGridLines = True

ws_dep.merge_cells("B2:I2")
d_hdr = ws_dep["B2"]
d_hdr.value = "CRITICAL PATH DEPENDENCY LOG & COTS VENDOR SCORECARD"
style_cell(d_hdr, font=Font(name=FONT_FAMILY, size=15, bold=True, color=COLOR_WHITE),
           fill=PatternFill("solid", fgColor=COLOR_NAVY),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_dep.row_dimensions[2].height = 36

ws_dep.merge_cells("B3:I3")
d_sub = ws_dep["B3"]
d_sub.value = "EXTERNAL BLOCKER ESCALATION REGISTER AND COMMERCIAL SOFTWARE EVALUATION MATRIX"
style_cell(d_sub, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN),
           fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_dep.row_dimensions[3].height = 20

# Section 1 Header: External Dependencies
ws_dep.merge_cells("B5:I5")
ws_dep["B5"].value = "SECTION 1: CRITICAL PATH EXTERNAL DEPENDENCY LOG (LEADERSHIP UNBLOCKING REQUIRED)"
style_cell(ws_dep["B5"], font=Font(name=FONT_FAMILY, size=10, bold=True, color=COLOR_WHITE),
           fill=PatternFill("solid", fgColor="DC2626"),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_dep.row_dimensions[5].height = 22

dep_headers = [
    ("B6", "Dep ID", 8, "center"),
    ("C6", "Impacted Project", 24, "left"),
    ("D6", "External Dependency Description", 32, "left"),
    ("E6", "External Provider / Team", 22, "left"),
    ("F6", "Impact Severity", 14, "center"),
    ("G6", "Current Status", 18, "center"),
    ("H6", "Days Waiting", 14, "center"),
    ("I6", "Actionable Request from Leadership / Director", 45, "left"),
]

for pos, h_title, col_w, align in dep_headers:
    cell = ws_dep[pos]
    cell.value = h_title
    style_cell(cell, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal=align, vertical="center"),
               border=border_header)
    ws_dep.column_dimensions[pos[0]].width = col_w

ws_dep.row_dimensions[6].height = 24

dependencies_data = [
    ("DEP-01", "Booking Platform (Website)", "Client Contractual Rate Sync API Endpoint", "Shipmnts Engineering Team", "Critical (P1)", "Waiting on API", "6 Days", "Facilitate engineering handshake with Shipmnts lead to release API specs & test sandbox."),
    ("DEP-02", "Booking Platform (Website)", "Post-Booking Workflow Job Creation API", "Shipmnts Engineering Team", "Critical (P1)", "Waiting on API", "6 Days", "Obtain confirmation on direct automated job dispatch vs email notification payload."),
    ("DEP-03", "Lead Generation Engine", "External Lead B2B Database (Apollo DB Export)", "Commercial Data Team / Tyra", "High (P2)", "Pending Handover", "7 Days", "Request export file handover from data team; Deepak is 100% ready to execute 1-day merge."),
    ("DEP-04", "Courier Pricing Engine", "Courier Official Rate Sheet with Approved Markup", "Commercial / Pricing Ops", "High (P2)", "Pending Tariff", "4 Days", "Provide signed-off courier tariff sheet; unlocks 5-day rapid development and cloud release."),
    ("DEP-05", "Tool Suggestion (Vendors)", "cargo.one Carrier Platform Enterprise Demo", "cargo.one Partner Team", "Medium (P3)", "Awaiting Response", "6 Days", "Reach out via executive commercial channel to expedite demo scheduling."),
    ("DEP-06", "Tool Suggestion (Vendors)", "RAFT Freight AI Operations Platform Demo", "RAFT Enterprise Sales", "Medium (P3)", "Awaiting Response", "5 Days", "Confirm discovery session date for logistics AI evaluation.")
]

for idx, (d_id, proj, desc, team, sev, stat, wait_d, act) in enumerate(dependencies_data, start=7):
    ws_dep[f"B{idx}"] = d_id
    style_cell(ws_dep[f"B{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"C{idx}"] = proj
    style_cell(ws_dep[f"C{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_NAVY), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_dep[f"D{idx}"] = desc
    style_cell(ws_dep[f"D{idx}"], font=Font(name=FONT_FAMILY, size=8.5), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_dep[f"E{idx}"] = team
    style_cell(ws_dep[f"E{idx}"], font=Font(name=FONT_FAMILY, size=8.5), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_dep[f"F{idx}"] = sev
    style_cell(ws_dep[f"F{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color="991B1B"), fill=PatternFill("solid", fgColor="FEE2E2"), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"G{idx}"] = stat
    style_cell(ws_dep[f"G{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color="92400E"), fill=PatternFill("solid", fgColor="FEF3C7"), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"H{idx}"] = wait_d
    style_cell(ws_dep[f"H{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"I{idx}"] = act
    style_cell(ws_dep[f"I{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_DARK_TEXT), alignment=Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1), border=border_thin)

    ws_dep.row_dimensions[idx].height = 24

# Section 2 Header: Vendor Evaluation Scorecard (Row 15)
ws_dep.merge_cells("B15:I15")
ws_dep["B15"].value = "SECTION 2: COMMERCIAL VENDOR EVALUATION SCORECARD & RECOMMENDATIONS"
style_cell(ws_dep["B15"], font=Font(name=FONT_FAMILY, size=10, bold=True, color=COLOR_WHITE),
           fill=PatternFill("solid", fgColor=COLOR_NAVY),
           alignment=Alignment(horizontal="left", vertical="center", indent=1))
ws_dep.row_dimensions[15].height = 22

vendor_headers = [
    ("B16", "Vendor Name", 16, "left"),
    ("C16", "Domain / Category", 22, "left"),
    ("D16", "Evaluation Status", 18, "center"),
    ("E16", "Engagement Date", 15, "center"),
    ("F16", "Key Strengths / Capabilities", 32, "left"),
    ("G16", "Identified Tradeoffs / Limitations", 30, "left"),
    ("H16", "Strategic Alignment Score", 16, "center"),
    ("I16", "Recommendation & Immediate Next Step", 40, "left"),
]

for pos, h_title, col_w, align in vendor_headers:
    cell = ws_dep[pos]
    cell.value = h_title
    style_cell(cell, font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_WHITE),
               fill=PatternFill("solid", fgColor=COLOR_NAVY_LIGHT),
               alignment=Alignment(horizontal=align, vertical="center"),
               border=border_header)

ws_dep.row_dimensions[16].height = 24

vendors_data = [
    ("Nexcade.ai", "AI Email Quote Parsing", "In Progress", "08 Oct 2026", "Automated email extraction; specialized logistics models.", "Enterprise pricing tier; second discovery call required.", "High (8.5/10)", "Complete second architecture call; benchmark against proprietary build with Priyank."),
    ("cargo.one", "Carrier Digital Booking", "Waiting on Dependency", "Demo Requested", "Extensive airline integrations; instant rate booking.", "Requires formal carrier commercial contracts.", "Medium (7.0/10)", "Deepak to follow up with vendor sales rep for European & Asian origin coverage."),
    ("CargoMart", "Freight Rate Procurement", "In Progress", "07 Oct 2026", "Comprehensive freight rate discovery & lane comparisons.", "Integration depth with existing TMS requires audit.", "High (8.0/10)", "Assess API flexibility for automated quoting ingestion into Booking Platform."),
    ("RAFT", "Logistics Operations AI", "Waiting on Dependency", "Demo Requested", "Strong customs & freight doc intelligence capabilities.", "Longer enterprise procurement cycle.", "Medium (7.5/10)", "Attend discovery call once confirmed by vendor team."),
    ("Zoho Infra & CRM", "Outbound Email & CRM", "Needs Decision", "Demo Complete", "All-in-one ecosystem; strong contact segmentation.", "Deliverability tuning and higher tier add-on fees.", "Very High (9.0/10)", "Finalize comparison with MailerCloud; make recommendation to Tyra."),
    ("MailerCloud", "High-Volume Email Delivery", "Needs Decision", "Demo Complete", "Exceptional deliverability rates; aggressive cost-per-email.", "Standalone mailer requiring custom CRM API integration.", "High (8.5/10)", "Evaluate total cost of ownership against Zoho suite.")
]

for idx, (v_name, dom, stat, dt, strg, lim, score, rec) in enumerate(vendors_data, start=17):
    ws_dep[f"B{idx}"] = v_name
    style_cell(ws_dep[f"B{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_NAVY), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    ws_dep[f"C{idx}"] = dom
    style_cell(ws_dep[f"C{idx}"], font=Font(name=FONT_FAMILY, size=8.5), alignment=Alignment(horizontal="left", vertical="center", indent=1), border=border_thin)

    st_cfg = STATUS_STYLES.get(stat, STATUS_STYLES["Planned"])
    ws_dep[f"D{idx}"] = stat
    style_cell(ws_dep[f"D{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=st_cfg["font"]),
               fill=PatternFill("solid", fgColor=st_cfg["fill"]),
               alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"E{idx}"] = dt
    style_cell(ws_dep[f"E{idx}"], font=Font(name=FONT_FAMILY, size=8.5), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"F{idx}"] = strg
    style_cell(ws_dep[f"F{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_DARK_TEXT), alignment=Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1), border=border_thin)

    ws_dep[f"G{idx}"] = lim
    style_cell(ws_dep[f"G{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_MUTED_TEXT), alignment=Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1), border=border_thin)

    ws_dep[f"H{idx}"] = score
    style_cell(ws_dep[f"H{idx}"], font=Font(name=FONT_FAMILY, size=8.5, bold=True, color=COLOR_CYAN_DARK), alignment=Alignment(horizontal="center", vertical="center"), border=border_thin)

    ws_dep[f"I{idx}"] = rec
    style_cell(ws_dep[f"I{idx}"], font=Font(name=FONT_FAMILY, size=8.5, color=COLOR_DARK_TEXT), alignment=Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1), border=border_thin)

    ws_dep.row_dimensions[idx].height = 24

# Save Workbook to both Downloads and Workspace
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
wb.save(OUTPUT_PATH)
wb.save(WORKSPACE_PATH)
print(f"SUCCESS: Workbook saved to {OUTPUT_PATH} and {WORKSPACE_PATH}")
