import json
import html
import urllib.parse
import zlib
import base64
import os

def esc(text):
    return html.escape(str(text), quote=True)

class DrawioDiagram:
    def __init__(self, title, width=1800, height=2300):
        self.title = title
        self.width = width
        self.height = height
        self.cells = []
        self.cell_id = 2

    def add_node(self, x, y, w, h, value, style):
        cid = str(self.cell_id)
        self.cell_id += 1
        xml = f'<mxCell id="{cid}" value="{esc(value)}" style="{style}" vertex="1" parent="1">\n'
        xml += f'  <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        xml += f'</mxCell>'
        self.cells.append(xml)
        return cid

    def add_edge(self, source_id, target_id, label="", style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#475569;strokeWidth=1.5;fontColor=#334155;fontSize=11;"):
        cid = str(self.cell_id)
        self.cell_id += 1
        val_attr = f'value="{esc(label)}"' if label else 'value=""'
        xml = f'<mxCell id="{cid}" {val_attr} style="{style}" edge="1" source="{source_id}" target="{target_id}" parent="1">\n'
        xml += f'  <mxGeometry relative="1" as="geometry" />\n'
        xml += f'</mxCell>'
        self.cells.append(xml)
        return cid

    def to_xml(self):
        body = "\n".join(self.cells)
        return f'''<mxfile host="app.diagrams.net">
<diagram id="workflow_diagram" name="{esc(self.title)}">
<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.width}" pageHeight="{self.height}" math="0" shadow="0">
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
{body}
  </root>
</mxGraphModel>
</diagram>
</mxfile>'''

# Styling
STYLE_TITLE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#0f172a;strokeColor=#1e293b;fontColor=#ffffff;fontSize=16;fontStyle=1;align=center;arcSize=6;"
STYLE_SUBTITLE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#334155;fontColor=#94a3b8;fontSize=11;align=center;arcSize=6;"
STYLE_SEC_BADGE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e2e8f0;strokeColor=#64748b;fontColor=#0f172a;fontSize=11;fontStyle=1;align=center;arcSize=8;"
STYLE_START = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;fontColor=#1e40af;fontSize=12;fontStyle=1;align=center;arcSize=20;"
STYLE_STEP = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#0284c7;fontColor=#0f172a;fontSize=12;fontStyle=0;align=center;arcSize=8;"
STYLE_DECISION = "rhombus;whiteSpace=wrap;html=1;fillColor=#fef3c7;strokeColor=#d97706;fontColor=#78350f;fontSize=11;fontStyle=1;align=center;"
STYLE_WARNING = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffedd5;strokeColor=#ea580c;fontColor=#9a3412;fontSize=12;fontStyle=1;align=center;arcSize=8;"
STYLE_DANGER = "rounded=1;whiteSpace=wrap;html=1;fillColor=#fee2e2;strokeColor=#dc2626;fontColor=#991b1b;fontSize=12;fontStyle=1;align=center;arcSize=8;"
STYLE_SUCCESS = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dcfce7;strokeColor=#16a34a;fontColor=#14532d;fontSize=11;fontStyle=1;align=center;arcSize=10;"
STYLE_NOTE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#fffdf5;strokeColor=#f59e0b;fontColor=#1c1917;fontSize=11;fontStyle=0;align=left;spacingLeft=14;spacingRight=14;spacingTop=10;spacingBottom=10;arcSize=6;"

EDGE_STD = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#475569;strokeWidth=1.5;fontColor=#334155;fontSize=11;"
EDGE_YES = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#16a34a;strokeWidth=2;fontColor=#16a34a;fontSize=11;fontStyle=1;"
EDGE_NO = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#dc2626;strokeWidth=2;fontColor=#dc2626;fontSize=11;fontStyle=1;"
EDGE_LOOP = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#ea580c;strokeWidth=2;fontColor=#ea580c;fontSize=11;fontStyle=1;dashed=1;"
EDGE_NOTE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#f59e0b;strokeWidth=1;dashed=1;endArrow=oval;endFill=1;startArrow=none;"

diag = DrawioDiagram("Air Export General Cargo & Billing Workflow", width=1780, height=2250)

# 1. Header Banner
diag.add_node(40, 30, 1700, 55,
    "<b>AIR EXPORT GENERAL CARGO & BILLING WORKFLOW</b><br/><font style='font-size:12px;font-weight:normal;'>Standard Operating Procedure: Dynamic Pricing, Software Entry, Operations Handover, 3-Way Reconciliation & Customer Invoicing</font>",
    STYLE_TITLE)

diag.add_node(40, 90, 1700, 32,
    "<b>Scope:</b> General Cargo Air Export &nbsp;|&nbsp; <b>Key Actors:</b> Customer / Agent, Pricing Team, Airlines, Booking Team, Operations, Billing Team &nbsp;|&nbsp; <b>Key Target:</b> Invoice Raised within 2 Days of Execution",
    STYLE_SUBTITLE)

# Layout:
# Left Section Badges: x=40, w=150
# Flow Steps: x=240, w=280
# Decisions: x=280, w=200, h=80
# Alternative branch nodes: x=580, w=250
# Right Notes: x=900, w=840

# ==========================================
# SECTION 1: PRE-SHIPMENT, RATES & SURCHARGES
# ==========================================
diag.add_node(40, 150, 160, 290,
    "<b>SECTION 1</b><br/><br/><b>Pre-Shipment, Rates & Airline Surcharges</b><br/><br/>• Rate Validity Span<br/>• Dynamic Negotiation<br/>• Carrier Matching<br/>• Weekly Fuel Surcharges<br/>• 1:1 Margin Protection",
    STYLE_SEC_BADGE)

s1_start = diag.add_node(240, 150, 280, 50, "<b>Start: Rate Inquiry / Booking Request</b><br/><font style='font-size:10px;'>Rate validity span defined (e.g., from 28th)</font>", STYLE_START)

s1_match = diag.add_node(240, 225, 280, 60, "<b>Airline Rate Matching & Negotiation</b><br/><font style='font-size:10px;'>Pricing Team coordinates directly with airline;<br/>Negotiations continue dynamically up to departure</font>", STYLE_STEP)
diag.add_edge(s1_start, s1_match)

s1_dec_fsc = diag.add_node(280, 310, 200, 80, "<b>Does Airline Increase<br/>Fuel Surcharge (FSC)?<br/>(e.g., Emirates Weekly)</b>", STYLE_DECISION)
diag.add_edge(s1_match, s1_dec_fsc)

s1_adjust = diag.add_node(580, 320, 260, 60, "<b>Adjust Selling Price Upward 1:1 (+Δ)</b><br/><font style='font-size:10px;'>If buy rate increases by 10, selling price<br/>must increase by 10 to protect profit margin</font>", STYLE_WARNING)
diag.add_edge(s1_dec_fsc, s1_adjust, label="Yes (Buy +Δ)", style=EDGE_YES)

s1_confirm = diag.add_node(240, 415, 280, 50, "<b>Confirm Approved Buy & Sell Rates</b><br/><font style='font-size:10px;'>Final rate pair confirmed before cargo dispatch</font>", STYLE_STEP)
diag.add_edge(s1_dec_fsc, s1_confirm, label="No (Rates Steady)", style=EDGE_NO)
diag.add_edge(s1_adjust, s1_confirm)

note_1 = diag.add_node(900, 150, 840, 160,
    "<b>EXPLANATORY NOTE 1: PRE-SHIPMENT, RATES & AIRLINE SURCHARGES</b><br/>"
    "• <b>Rate Validity & Dynamic Negotiations:</b> Rate validity can span across specific dates (e.g., starting from the 28th). Rate negotiations and matching with airline rates continue dynamically right up until the shipment actually departs.<br/>"
    "• <b>Airline Rate Matching:</b> The pricing team coordinates closely to match and confirm rates directly with the airline before dispatch.<br/>"
    "• <b>Managing Fuel Surcharges (FSC):</b> Carriers like Emirates update fuel surcharges almost weekly. If an airline increases its fuel surcharge (e.g., buy rate increases by 10), the selling price must be adjusted upward by the exact same amount (e.g., selling price increases by 10) to ensure your profit margin remains protected.",
    STYLE_NOTE)
diag.add_edge(s1_match, note_1, style=EDGE_NOTE)


# ==========================================
# SECTION 2: BOOKING AND SYSTEM ENTRY
# ==========================================
diag.add_node(40, 480, 160, 220,
    "<b>SECTION 2</b><br/><br/><b>Booking & System Entry</b><br/><br/>• Customer Confirmation<br/>• Software Integration<br/>• RFQ to Booking<br/>• System Quotation (Buy 100 / Sell 110)",
    STYLE_SEC_BADGE)

s2_cust_confirm = diag.add_node(240, 490, 280, 50, "<b>Customer Confirms Booking</b><br/><font style='font-size:10px;'>Shipper awards shipment based on confirmed terms</font>", STYLE_START)
diag.add_edge(s1_confirm, s2_cust_confirm)

s2_sw_entry = diag.add_node(240, 565, 280, 60, "<b>Software Integration: Move RFQ to Booking</b><br/><font style='font-size:10px;'>Booking Team moves record from RFQ to Booking<br/>inside the shipment management software</font>", STYLE_STEP)
diag.add_edge(s2_cust_confirm, s2_sw_entry)

s2_quote = diag.add_node(240, 650, 280, 65, "<b>Generate System Quotation</b><br/><font style='font-size:10px;'>Generates system quotation using Pricing Team's<br/>approved rates (e.g., buy rate 100, sell rate 110)</font>", STYLE_STEP)
diag.add_edge(s2_sw_entry, s2_quote)

note_2 = diag.add_node(900, 500, 840, 140,
    "<b>EXPLANATORY NOTE 2: BOOKING AND SYSTEM ENTRY</b><br/>"
    "• <b>Initial Booking:</b> The process starts when a customer confirms a booking.<br/>"
    "• <b>Software Integration:</b> The booking team moves the record from RFQ to booking inside the shipment software and generates a system quotation using the pricing team's approved rates (e.g., buy rate of 100 and sell rate of 110).<br/>"
    "• <b>Single Source of Truth:</b> The system quotation binds the authorized commercial figures to prevent unauthorized rate drift.",
    STYLE_NOTE)
diag.add_edge(s2_quote, note_2, style=EDGE_NOTE)


# ==========================================
# SECTION 3: EXECUTION & HANDOVER
# ==========================================
diag.add_node(40, 740, 160, 360,
    "<b>SECTION 3</b><br/><br/><b>Execution & Handover</b><br/><br/>• Operations Oversight<br/>• Flight Schedules<br/>• Handover Cutoffs<br/>• Customs Statuses<br/>• Pure Forwarding<br/>• Billing Request Email",
    STYLE_SEC_BADGE)

s3_ops = diag.add_node(240, 740, 280, 65, "<b>Operations Execution Management</b><br/><font style='font-size:10px;'>Pricing Team manages execution, tracking flight schedules,<br/>handover times, and customs clearance statuses</font>", STYLE_STEP)
diag.add_edge(s2_quote, s3_ops)

s3_dec_pure = diag.add_node(280, 830, 200, 80, "<b>Is Shipment<br/>Pure Forwarding?</b>", STYLE_DECISION)
diag.add_edge(s3_ops, s3_dec_pure)

s3_direct_handover = diag.add_node(580, 840, 260, 60, "<b>Proceed Directly to Handover</b><br/><font style='font-size:10px;'>Pure forwarding shipments bypass brokerage<br/>and proceed directly to airline handover</font>", STYLE_STEP)
diag.add_edge(s3_dec_pure, s3_direct_handover, label="Yes (Pure Forwarding)", style=EDGE_YES)

s3_customs = diag.add_node(240, 935, 280, 55, "<b>Complete Customs Clearance & Handover</b><br/><font style='font-size:10px;'>Customs cleared -> Handover to airline cargo terminal</font>", STYLE_STEP)
diag.add_edge(s3_dec_pure, s3_customs, label="No (Standard)", style=EDGE_NO)

s3_flight = diag.add_node(240, 1015, 280, 50, "<b>Flight Departure (Execution Date)</b><br/><font style='font-size:10px;'>Shipment departs; execution date officially recorded</font>", STYLE_START)
diag.add_edge(s3_customs, s3_flight)
diag.add_edge(s3_direct_handover, s3_flight)

s3_bill_req = diag.add_node(240, 1090, 280, 75, "<b>Pricing Team Initiates Billing Request</b><br/><font style='font-size:10px;'>Using <b>execution date</b> as reference, Pricing Team emails<br/>Billing Team via <b>customer communication thread</b>,<br/>citing the agreed buy and sell rates</font>", STYLE_STEP)
diag.add_edge(s3_flight, s3_bill_req)

note_3 = diag.add_node(900, 800, 840, 160,
    "<b>EXPLANATORY NOTE 3: EXECUTION & HANDOVER</b><br/>"
    "• <b>Operations Management:</b> The pricing team manages shipment execution, tracking flight schedules, handover times, and customs clearance statuses. Pure forwarding shipments proceed directly to handover.<br/>"
    "• <b>Billing Request Initiation:</b> Using the execution date as a reference, the pricing team initiates the billing request by emailing the billing team via the customer communication thread, citing the agreed buy and sell rates.<br/>"
    "• <b>Audit Trail Continuity:</b> Communicating on the customer thread maintains full visibility into rate approvals and client instructions.",
    STYLE_NOTE)
diag.add_edge(s3_bill_req, note_3, style=EDGE_NOTE)


# ==========================================
# SECTION 4: VERIFICATION, DISCREPANCIES & FEEDBACK
# ==========================================
diag.add_node(40, 1200, 160, 260,
    "<b>SECTION 4</b><br/><br/><b>Verification, Discrepancies & Feedback</b><br/><br/>• Cross-Tallying Rates<br/>• Error Identification<br/>• Correction Feedback Loop<br/>• Resolution Before Invoicing",
    STYLE_SEC_BADGE)

s4_tally = diag.add_node(240, 1200, 280, 75, "<b>Billing Team Cross-Tallying</b><br/><font style='font-size:10px;'>Billing Team cross-verifies: (1) System Quotation,<br/>(2) Pricing Team's Email Request, and (3) Airline Rates</font>", STYLE_STEP)
diag.add_edge(s3_bill_req, s4_tally)

s4_dec_error = diag.add_node(280, 1305, 200, 80, "<b>Any Mistakes,<br/>Discrepancies, or<br/>Requested Changes?</b>", STYLE_DECISION)
diag.add_edge(s4_tally, s4_dec_error)

s4_feedback = diag.add_node(580, 1310, 260, 70, "<b>Correction Loop: Reply with Feedback</b><br/><font style='font-size:10px;'>Billing Team replies with feedback outlining necessary<br/>corrections before invoice is finalized</font>", STYLE_DANGER)
diag.add_edge(s4_dec_error, s4_feedback, label="Yes (Discrepancy)", style=EDGE_YES)

s4_fix = diag.add_node(580, 1205, 260, 65, "<b>Pricing / Booking Team Corrections</b><br/><font style='font-size:10px;'>Applies required corrections to system quotation<br/>or clarifies rates on thread</font>", STYLE_WARNING)
diag.add_edge(s4_feedback, s4_fix)
diag.add_edge(s4_fix, s4_tally, label="Re-tally", style=EDGE_LOOP)

s4_approved = diag.add_node(240, 1415, 280, 50, "<b>Rates & Charges Cross-Verified</b><br/><font style='font-size:10px;'>Clean verification confirmed; ready for invoicing</font>", STYLE_SUCCESS)
diag.add_edge(s4_dec_error, s4_approved, label="No (Clean Tally)", style=EDGE_NO)

note_4 = diag.add_node(900, 1200, 840, 160,
    "<b>EXPLANATORY NOTE 4: VERIFICATION, DISCREPANCIES & FEEDBACK</b><br/>"
    "• <b>Cross-Tallying:</b> The billing team cross-verifies the system quotation against the pricing team's email request and airline rates.<br/>"
    "• <b>Correction Loops:</b> If the billing team identifies any mistakes, discrepancies, or requested changes, they reply with feedback outlining the necessary corrections before the invoice is finalized.<br/>"
    "• <b>Pre-Billing Control:</b> No invoice is generated while discrepancies or rate variances remain unresolved.",
    STYLE_NOTE)
diag.add_edge(s4_feedback, note_4, style=EDGE_NOTE)


# ==========================================
# SECTION 5: INVOICING, DRAFTS & DOCUMENT DELIVERY
# ==========================================
diag.add_node(40, 1500, 160, 480,
    "<b>SECTION 5</b><br/><br/><b>Invoicing, Drafts, and Document Delivery</b><br/><br/>• Multi-Location Agents<br/>• 24h Draft Approval<br/>• Rework Minimization<br/>• Digital Soft Copy (Default)<br/>• Monthly Hard Copy<br/>• Strict 2-Day Target",
    STYLE_SEC_BADGE)

s5_dec_agent = diag.add_node(280, 1500, 200, 80, "<b>Is Client a<br/>Multi-Location Agent<br/>Needing Revisions?</b>", STYLE_DECISION)
diag.add_edge(s4_approved, s5_dec_agent)

s5_draft = diag.add_node(580, 1505, 260, 70, "<b>Share Draft Invoice within 24 Hours</b><br/><font style='font-size:10px;'>Draft invoice shared for approval within <b>24 hours</b><br/>for address adjustments/revisions to minimize rework</font>", STYLE_WARNING)
diag.add_edge(s5_dec_agent, s5_draft, label="Yes (Multi-Location Agent)", style=EDGE_YES)

s5_final = diag.add_node(240, 1615, 280, 60, "<b>Raise Final Tax Invoice</b><br/><font style='font-size:10px;'>Final invoice generated per approved rates/address<br/><b>Target: Within 2 days of shipment execution</b></font>", STYLE_STEP)
diag.add_edge(s5_dec_agent, s5_final, label="No (Standard)", style=EDGE_NO)
diag.add_edge(s5_draft, s5_final, label="Approved")

s5_dec_format = diag.add_node(280, 1710, 200, 80, "<b>Does Client Specifically<br/>Require Physical<br/>Hard Copy?</b>", STYLE_DECISION)
diag.add_edge(s5_final, s5_dec_format)

s5_hard = diag.add_node(580, 1715, 260, 70, "<b>Monthly Physical Dispatch Batch</b><br/><font style='font-size:10px;'>If client specifically requires physical hard copy,<br/>they are dispatched on a <b>monthly basis</b></font>", STYLE_STEP)
diag.add_edge(s5_dec_format, s5_hard, label="Yes (Physical Required)", style=EDGE_YES)

s5_soft = diag.add_node(240, 1825, 280, 65, "<b>Digital Email Delivery (Soft Copy)</b><br/><font style='font-size:10px;'>Invoices are primarily handled digitally via email<br/>using soft copies within 2 days of execution</font>", STYLE_SUCCESS)
diag.add_edge(s5_dec_format, s5_soft, label="No (Digital Default)", style=EDGE_NO)
diag.add_edge(s5_hard, s5_soft)

s5_end = diag.add_node(240, 1920, 280, 50, "<b>Workflow Completed</b><br/><font style='font-size:10px;'>Invoice successfully dispatched to customer</font>", STYLE_START)
diag.add_edge(s5_soft, s5_end)

note_5 = diag.add_node(900, 1550, 840, 190,
    "<b>EXPLANATORY NOTE 5: INVOICING, DRAFTS, AND DOCUMENT DELIVERY</b><br/>"
    "• <b>Invoicing Format:</b> Invoices are primarily handled digitally via email using soft copies. However, if a client specifically requires a physical hard copy, they are dispatched on a <b>monthly basis</b>.<br/>"
    "• <b>Draft Approvals:</b> For multi-location agent clients who frequently need address adjustments or revisions, draft invoices are shared for approval within <b>24 hours</b> to minimize later rework.<br/>"
    "• <b>Timeline Targets:</b> The core performance target is to successfully raise and dispatch the final invoice within <b>two days</b> of shipment execution.",
    STYLE_NOTE)
diag.add_edge(s5_final, note_5, style=EDGE_NOTE)

xml_content = diag.to_xml()

# Save standalone .drawio file
out_path_1 = "/Users/deepak/Downloads/air_export_general_cargo_and_billing_workflow.drawio"
out_path_2 = "/Users/deepak/Downloads/SOP/Air_Export_General_Cargo_and_Billing/air_export_general_cargo_and_billing_workflow.drawio"

with open(out_path_1, "w") as f:
    f.write(xml_content)

with open(out_path_2, "w") as f:
    f.write(xml_content)

# Generate direct online editor URL
compressed = zlib.compress(xml_content.encode('utf-8'))[2:-4]
b64 = base64.b64encode(compressed).decode('utf-8')
payload = {'type': 'xml', 'compressed': True, 'data': b64}
encoded = urllib.parse.quote(json.dumps(payload))
url = f"https://app.diagrams.net/?grid=0&pv=0&border=10&edit=_blank#create={encoded}"

with open("/Users/deepak/Downloads/air_export_workflow_url.txt", "w") as uf:
    uf.write(url)

print(f"Generated standalone workflow file successfully: {out_path_1} ({len(xml_content)} bytes)")
print(f"Direct Draw.io URL length: {len(url)}")
print(f"URL: {url[:100]}...")
