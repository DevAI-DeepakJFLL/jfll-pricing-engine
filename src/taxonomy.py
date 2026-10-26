"""
src/taxonomy.py
Enterprise Commodity Classification and Geographic Region Resolution.

Fixes:
- Issue C1: Substring keyword collisions replaced with compiled regex word boundaries (\\b).
- Issue C2: Dest-region short keyword collisions replaced with country-first resolution and word boundaries.
- Issue D7: Restores Dangerous Goods, Valuables, Live Animals, and proper 'Other' handling.
"""

import re
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------------------
# CANONICAL VOCABULARIES
# ------------------------------------------------------------------------------

COMMODITY_CATEGORIES: List[str] = [
    "General Cargo",
    "Dangerous Goods",
    "Valuables",
    "Live Animals",
    "Pharmaceuticals",
    "Perishable Foodstuff",
    "Auto Parts",
    "Engineering & Machinery",
    "Garments / Textiles",
    "Courier",
    "Other",
]

DESTINATION_REGIONS: List[str] = [
    "Europe",
    "Middle East",
    "North America",
    "Asia-Pacific",
    "Latin America",
    "Africa",
    "Indian Subcontinent",
]

CARRIER_TIERS: List[str] = [
    "Premium",
    "Standard",
    "Budget",
]

CARRIER_TIER_PREMIUM: List[str] = [
    "Emirates",
    "Qatar Airways",
    "Singapore Airlines",
    "Cathay Pacific",
    "Lufthansa",
    "Etihad Airways",
    "British Airways",
]

CARRIER_TIER_STANDARD: List[str] = [
    "Air India",
    "Ethiopian Airlines",
    "Air France",
    "Turkish Airlines",
    "Saudi Airlines",
    "SriLankan Airlines",
    "Gulf Air",
    "Oman Air",
    "EgyptAir",
    "Kuwait Airways",
]

CARRIER_TIER_BUDGET: List[str] = [
    "IndiGo",
    "SpiceJet",
    "Air Arabia",
    "Flydubai",
]


# ------------------------------------------------------------------------------
# COMPILED WORD-BOUNDARY REGEX PATTERNS (COMMODITIES)
# ------------------------------------------------------------------------------

# 1. Negative DG Guard: strings explicitly stating non-dangerous status
RE_NON_DG = re.compile(
    r"\b(?:NON[\s-]*(?:DG|DGR|HAZ|HAZARDOUS)|NOT\s+RESTRICTED|NOT\s+DANGEROUS)\b",
    re.IGNORECASE,
)

# 2. Dangerous Goods (UN numbers, hazmat classes, lithium batteries, etc.)
# Word boundary ensures plain 'UN' or generic 'CHEMICAL' does NOT match.
RE_UN_NUMBER = re.compile(r"\bUN\s*\d{4}\b", re.IGNORECASE)
RE_DG_SPECIFIC = re.compile(
    r"\b(?:"
    r"DG|DGR|HAZMAT|DANGEROUS\s+GOODS|"
    r"CLASS\s*[1-9](?:\.[0-9])?|"
    r"HAZARDOUS(?:\s+CARGO|\s+MATERIAL|\s+CHEMICAL|\s+GOODS)?|"
    r"LITHIUM(?:\s+(?:ION|METAL))?\s+BATTER(?:Y|IES)|BATTER(?:Y|IES)\s+LITHIUM|"
    r"FLAMMABLE|RADIOACTIVE|EXPLOSIVE|CORROSIVE|BIOHAZARD"
    r")\b",
    re.IGNORECASE,
)

# 3. Valuables (Gold, Silver, Gems, Currency, Watches)
RE_VALUABLES = re.compile(
    r"\b(?:"
    r"VALUABLES?|VAL|VUN|"
    r"GOLD|SILVER|JEWELLERY|JEWELRY|JEWELS?|GEMS?|GEMSTONES?|"
    r"DIAMONDS?|PRECIOUS\s+STONES?|PRECIOUS\s+METALS?|"
    r"CURRENCY|BANKNOTES?|BULLION|"
    r"WATCH(?:ES)?|WRIST\s*WATCH(?:ES)?"
    r")\b",
    re.IGNORECASE,
)

# 4. Live Animals (PET, AVI, Dogs, Cats, Horses, Day old chicks, Live/Ornamental fish)
# Strict word boundaries: \bPET\b prevents petroleum/carpet; \bAVI\b prevents aviation/avionics.
RE_LIVE_ANIMALS = re.compile(
    r"\b(?:"
    r"LIVE\s+ANIMALS?|AVI|PETS?|"
    r"DOGS?|CATS?|HORSES?|"
    r"DAY\s+OLD\s+CHICKS?|CHICKS?|"
    r"ORNAMENTAL\s+(?:TROPICAL\s+)?FISH|LIVE\s+FISH|"
    r"EQUINE|CANINE|FELINE|LIVESTOCK|PUPPY|PUPPIES|KITTEN|KITTENS|BIRDS?|REPTILES?"
    r")\b",
    re.IGNORECASE,
)

# 5. Pharmaceuticals (Vaccines, medicines, diagnostic kits, pharma)
# Scoped so tablet computers / graphics tablets do NOT match.
RE_PHARMACEUTICALS = re.compile(
    r"\b(?:"
    r"PHARMA|PHARMACEUTICALS?|MEDICINES?|MEDICATIONS?|DRUGS?|"
    r"VACCINES?|INSULIN|SERUM|BIOLOGICALS?|DIAGNOSTIC\s+KITS?|"
    r"HEALTHCARE|MEDICAL\s+DEVICES?|CLINICAL\s+TRIALS?|ANTIBIOTICS?|"
    r"(?<!GRAPHICS\s)(?<!DIGITAL\s)(?<!DRAWING\s)TABLETS?(?!\s+(?:COMPUTER|PC|DEVICE))|"
    r"CAPSULES?|SYRUP"
    r")\b",
    re.IGNORECASE,
)

# 6. Perishable Foodstuff (Fresh fruit, mangoes, seafood, meat, cut flowers)
# Restricts handling code; removes bare \bPER\b to avoid 'AS PER INVOICE/PACKING LIST' collisions.
RE_PERISHABLES = re.compile(
    r"\b(?:"
    r"PERISHABLES?|"
    r"(?:SPL:\s*)?PER\s+CARGO|"
    r"FOOD|FOODSTUFFS?|FRUITS?|VEGETABLES?|VEG|"
    r"MEAT|FRESH\s+FISH|SEAFOOD|CRABS?|LOBSTERS?|PRAWNS?|SHRIMPS?|"
    r"MANGOES?|FLOWERS?|CUT\s+FLOWERS?|FLORAL|DAIRY|CHEESE|PRODUCE|"
    r"CHILLED|FROZEN\s+FOOD|FROZEN\s+MEAT|ASPARAGUS|BERRIES|BERRY"
    r")\b",
    re.IGNORECASE,
)

# 7. Auto Parts (Brake pads, transmission, tires, auto accessories)
# Strictly scoped to automotive contexts; bare 'SPARE PARTS' routes to Engineering & Machinery.
RE_AUTO_PARTS = re.compile(
    r"\b(?:"
    r"AUTO(?:MOBILE)?\s+(?:SPARE\s+)?PARTS?|"
    r"AUTOMOTIVE(?:\s+SPARE)?\s+PARTS?|"
    r"CAR\s+(?:SPARE\s+)?PARTS?|MOTOR\s+PARTS?|VEHICLE\s+(?:SPARE\s+)?PARTS?|"
    r"AUTO\s+ACCESSOR(?:Y|IES)|VEHICLE\s+ACCESSOR(?:Y|IES)|"
    r"AUTO\s+COMPONENTS?|BRAKE\s+PADS?|CLUTCH|TRANSMISSION(?:\s+PARTS?)?|"
    r"SHOCK\s+ABSORBERS?|TIRES?|TYRES?|CAR\s+ENGINE|"
    r"AUTO"
    r")\b",
    re.IGNORECASE,
)

# 8. Garments & Textiles (Fabrics, cotton, carpets, apparel, leather footwear)
# Excludes optical/carbon fiber which belongs in Engineering & Machinery.
RE_GARMENTS = re.compile(
    r"\b(?:"
    r"GARMENTS?|TEXTILES?|APPAREL|CLOTHING|FABRICS?|COTTON|YARN|SILK|LEATHER|"
    r"CARPETS?|RUGS?|WOVEN|KNITTED|FOOTWEAR|SHOES?|DENIM|"
    r"(?<!OPTICAL\s)(?<!CARBON\s)FIB(?:ER|RE)(?!\s*OPTIC)|"
    r"SHIRTS?|TROUSERS?|LINEN"
    r")\b",
    re.IGNORECASE,
)

# 9. Engineering & Machinery (Turbines, pumps, valves, machinery, automation, circuit boards)
# Captures industrial/aviation spare parts, optical/carbon fiber, and computing tablets.
RE_ENGINEERING = re.compile(
    r"\b(?:"
    r"ENGINEERING|MACHINERY|MACHINES?|EQUIPMENT|HARDWARE|"
    r"TURBINES?|PUMPS?|GENERATORS?|CIRCUIT\s+BOARDS?|VALVES?|TRANSFORMERS?|"
    r"COMPRESSORS?|INDUSTRIAL|ELECTRONICS?|MECHANICAL|TOOLS?|TOOLING|"
    r"MOTORS?|AUTOMATION|ENGINES?|RADAR|SENSORS?|BOILERS?|"
    r"(?:AVIATION|AIRCRAFT|AEROSPACE|MACHINE|MACHINERY|PUMP|INDUSTRIAL|TURBINE|GENERATOR|MECHANICAL|ENGINEERING)\s+SPARE\s+PARTS?|"
    r"SPARE\s+PARTS?|"
    r"OPTICAL\s+FIB(?:ER|RE)(?:\s+CABLES?)?|FIB(?:ER|RE)\s+OPTIC(?:S)?|CARBON\s+FIB(?:ER|RE)(?:\s+PANELS?|\s+COMPOSITES?)?|"
    r"TABLET\s+COMPUTER(?:S)?|GRAPHICS\s+TABLET(?:S)?|DIGITAL\s+TABLET(?:S)?"
    r")\b",
    re.IGNORECASE,
)

# 10. Courier & Express (Documents, samples, pouches)
RE_COURIER = re.compile(
    r"\b(?:"
    r"COURIER|EXPRESS|DOCUMENTS?|SAMPLES?|DIPLOMATIC\s+POUCH|POUCH|PARCELS?|LETTERS?"
    r")\b",
    re.IGNORECASE,
)

# 11. Explicit Other / Special Cargo
RE_OTHER = re.compile(
    r"\b(?:OTHER|MISCELLANEOUS|SPECIAL\s+CARGO|SPECIAL\s+GOODS|UNCLASSIFIED)\b",
    re.IGNORECASE,
)

# 12. Explicit General Cargo
RE_GENERAL_CARGO = re.compile(
    r"\b(?:"
    r"GENERAL\s+CARGO|GENERAL|GEN\s+CARGO|CONSOLIDATION|CONSOL|"
    r"FREIGHT\s+ALL\s+KINDS|FAK|MIXED\s+CARGO|MERCHANDISE|"
    r"PERSONAL\s+EFFECTS|HOUSEHOLD\s+GOODS"
    r")\b",
    re.IGNORECASE,
)


def classify_commodity(description: Any) -> str:
    """
    Classify a cargo description into canonical commodity categories using
    compiled regex patterns with word boundaries (\\b) to avoid substring collisions.

    Restores:
      - Dangerous Goods (DG, Hazmat, Class 1-9, UN numbers, Lithium Batteries)
      - Valuables (Gold, Silver, Jewellery, Gems, Currency, Watches)
      - Live Animals (Dogs, Cats, Horses, Day Old Chicks, Ornamental Fish)
      - Pharmaceuticals, Perishable Foodstuff, Auto Parts, Engineering, Garments, Courier.
      - Fallback: General Cargo or Other.
    """
    if description is None:
        return "General Cargo"

    desc = str(description).strip()
    if not desc:
        return "General Cargo"

    is_non_dg = bool(RE_NON_DG.search(desc))

    # 1. Dangerous Goods (guarded against NON-DG negations)
    if not is_non_dg:
        if RE_UN_NUMBER.search(desc) or RE_DG_SPECIFIC.search(desc):
            return "Dangerous Goods"

    # 2. Valuables
    if RE_VALUABLES.search(desc):
        return "Valuables"

    # 3. Live Animals
    if RE_LIVE_ANIMALS.search(desc):
        return "Live Animals"

    # 4. Pharmaceuticals
    if RE_PHARMACEUTICALS.search(desc):
        return "Pharmaceuticals"

    # 5. Perishables
    if RE_PERISHABLES.search(desc):
        return "Perishable Foodstuff"

    # 6. Auto Parts
    if RE_AUTO_PARTS.search(desc):
        return "Auto Parts"

    # 7. Garments / Textiles (checked before Engineering to catch carpets/textiles)
    if RE_GARMENTS.search(desc):
        return "Garments / Textiles"

    # 8. Engineering & Machinery
    if RE_ENGINEERING.search(desc):
        return "Engineering & Machinery"

    # 9. Courier
    if RE_COURIER.search(desc):
        return "Courier"

    # 10. Explicit Other
    if RE_OTHER.search(desc):
        return "Other"

    # 11. Explicit General Cargo
    if RE_GENERAL_CARGO.search(desc):
        return "General Cargo"

    # Default fallback for unclassified general cargo
    return "General Cargo"


# ------------------------------------------------------------------------------
# COMPILED WORD-BOUNDARY REGEX PATTERNS (CARRIERS)
# ------------------------------------------------------------------------------

RE_CARRIER_BUDGET = re.compile(
    r"\b(?:"
    r"INDIGO|6E|"
    r"SPICEJET|SPICE\s+JET|SG|"
    r"AIR\s+ARABIA(?:\s+ABU\s+DHABI)?|G9|"
    r"FLYDUBAI|FLY\s+DUBAI|FZ"
    r")\b",
    re.IGNORECASE,
)

RE_CARRIER_PREMIUM = re.compile(
    r"\b(?:"
    r"EMIRATES(?:\s+SKYCARGO)?|EK|"
    r"QATAR(?:\s+AIRWAYS)?|QR|"
    r"SINGAPORE\s+AIRLINES(?:\s+LTD)?|SIA|SQ|"
    r"CATHAY(?:\s+PACIFIC)?|CX|"
    r"LUFTHANSA(?:\s+CARGO)?|LH|"
    r"ETIHAD(?:\s+AIRWAYS)?|EY|"
    r"BRITISH\s+AIRWAYS|BA"
    r")\b",
    re.IGNORECASE,
)

RE_CARRIER_STANDARD = re.compile(
    r"\b(?:"
    r"AIR\s+INDIA|AI|"
    r"ETHIOPIAN(?:\s+AIRLINES)?|ET|"
    r"AIR\s+FRANCE|AF|"
    r"TURKISH(?:\s+AIRLINES)?|TK|"
    r"SAUDI(?:\s+AIRLINES)?(?:\s+CARGO)?|SAUDIA|SV|"
    r"SRILANKAN(?:\s+AIRLINES)?|UL|"
    r"GULF\s+AIR|GF|"
    r"OMAN\s+AIR|WY|"
    r"EGYPTAIR|EGYPT\s+AIR|MS|"
    r"KUWAIT(?:\s+AIRWAYS)?|KU"
    r")\b",
    re.IGNORECASE,
)


def classify_carrier_tier(carrier_name: Any) -> str:
    """
    Classify airline carrier into canonical commercial tiers:
    'Premium', 'Standard', or 'Budget'.

    - Premium: Emirates, Qatar Airways, Singapore Airlines, Cathay Pacific,
               Lufthansa, Etihad Airways, British Airways.
    - Budget: IndiGo, SpiceJet, Air Arabia, Flydubai.
    - Standard: Air India, Ethiopian Airlines, Air France, Turkish Airlines,
                Saudi Airlines, SriLankan Airlines, Gulf Air, Oman Air,
                EgyptAir, Kuwait Airways, and fallback for other commercial carriers.
    """
    if carrier_name is None:
        return "Standard"

    name_clean = str(carrier_name).strip()
    if not name_clean:
        return "Standard"

    name_upper = name_clean.upper()
    if name_upper in ("PREMIUM", "STANDARD", "BUDGET"):
        return name_upper.capitalize()

    # 1. Budget Carriers
    if RE_CARRIER_BUDGET.search(name_clean):
        return "Budget"

    # 2. Premium Carriers
    if RE_CARRIER_PREMIUM.search(name_clean):
        return "Premium"

    # 3. Standard Carriers
    if RE_CARRIER_STANDARD.search(name_clean):
        return "Standard"

    # Default fallback for commercial scheduled carriers
    return "Standard"


# ------------------------------------------------------------------------------
# GLOBAL COUNTRY TO FREIGHT CORRIDOR CATALOG
# ------------------------------------------------------------------------------

COUNTRY_TO_REGION: Dict[str, str] = {
    # Europe
    "United Kingdom": "Europe", "UK": "Europe", "Great Britain": "Europe", "England": "Europe",
    "Scotland": "Europe", "Wales": "Europe", "Northern Ireland": "Europe",
    "Germany": "Europe", "Netherlands": "Europe", "Holland": "Europe", "France": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Belgium": "Europe", "Switzerland": "Europe",
    "Sweden": "Europe", "Poland": "Europe", "Austria": "Europe", "Czech Republic": "Europe",
    "Czechia": "Europe", "Denmark": "Europe", "Ireland": "Europe", "Norway": "Europe",
    "Portugal": "Europe", "Finland": "Europe", "Greece": "Europe", "Hungary": "Europe",
    "Romania": "Europe", "Russian Federation": "Europe", "Russia": "Europe", "Turkey": "Europe",
    "Ukraine": "Europe", "Belarus": "Europe", "Bulgaria": "Europe", "Croatia": "Europe",
    "Slovakia": "Europe", "Slovenia": "Europe", "Lithuania": "Europe", "Latvia": "Europe",
    "Estonia": "Europe", "Luxembourg": "Europe", "Cyprus": "Europe", "Malta": "Europe",
    "Iceland": "Europe", "Albania": "Europe", "Bosnia and Herzegovina": "Europe",
    "North Macedonia": "Europe", "Macedonia": "Europe", "Serbia": "Europe", "Montenegro": "Europe",
    "Armenia": "Europe", "Georgia": "Europe", "Azerbaijan": "Europe", "Moldova": "Europe",
    "Monaco": "Europe", "Andorra": "Europe", "San Marino": "Europe", "Liechtenstein": "Europe",

    # Middle East
    "United Arab Emirates": "Middle East", "UAE": "Middle East", "Oman": "Middle East",
    "Qatar": "Middle East", "Saudi Arabia": "Middle East", "KSA": "Middle East",
    "Bahrain": "Middle East", "Kuwait": "Middle East", "Jordan": "Middle East",
    "Lebanon": "Middle East", "Iraq": "Middle East", "Israel": "Middle East",
    "Yemen": "Middle East", "Iran": "Middle East", "Iran, Islamic Republic of": "Middle East",
    "Syria": "Middle East", "Palestine": "Middle East",

    # North America
    "United States": "North America", "USA": "North America", "United States of America": "North America",
    "US": "North America", "Canada": "North America", "Mexico": "North America",
    "Bermuda": "North America", "Greenland": "North America",

    # Latin America
    "Brazil": "Latin America", "Argentina": "Latin America", "Chile": "Latin America",
    "Colombia": "Latin America", "Peru": "Latin America", "Ecuador": "Latin America",
    "Uruguay": "Latin America", "Paraguay": "Latin America", "Panama": "Latin America",
    "Costa Rica": "Latin America", "Guatemala": "Latin America", "Honduras": "Latin America",
    "Dominican Republic": "Latin America", "Jamaica": "Latin America",
    "Trinidad and Tobago": "Latin America", "Bolivia": "Latin America",
    "Bolivia, Plurinational State of": "Latin America", "El Salvador": "Latin America",
    "Nicaragua": "Latin America", "Venezuela": "Latin America",
    "Venezuela, Bolivarian Republic of": "Latin America", "Antigua and Barbuda": "Latin America",
    "Barbados": "Latin America", "Curaçao": "Latin America", "Curacao": "Latin America",
    "Guyana": "Latin America", "Suriname": "Latin America", "Haiti": "Latin America",
    "Bahamas": "Latin America", "Belize": "Latin America", "Saint Lucia": "Latin America",
    "Cuba": "Latin America", "Aruba": "Latin America", "Cayman Islands": "Latin America",
    "Guadeloupe": "Latin America", "Martinique": "Latin America",
    "Saint Kitts and Nevis": "Latin America", "Saint Vincent and the Grenadines": "Latin America",
    "Grenada": "Latin America",

    # Africa
    "Egypt": "Africa", "South Africa": "Africa", "Kenya": "Africa", "Nigeria": "Africa",
    "Ghana": "Africa", "Tanzania": "Africa", "Tanzania, United Republic of": "Africa",
    "Uganda": "Africa", "Ethiopia": "Africa", "Morocco": "Africa", "Algeria": "Africa",
    "Tunisia": "Africa", "Mauritius": "Africa", "Seychelles": "Africa", "Zambia": "Africa",
    "Zimbabwe": "Africa", "Rwanda": "Africa", "Mozambique": "Africa", "Angola": "Africa",
    "Senegal": "Africa", "Cote d'Ivoire": "Africa", "Ivory Coast": "Africa",
    "Cameroon": "Africa", "Namibia": "Africa", "Botswana": "Africa", "Madagascar": "Africa",
    "Malawi": "Africa", "Mali": "Africa", "Sudan": "Africa", "South Sudan": "Africa",
    "Congo": "Africa", "Congo, The Democratic Republic of the": "Africa",
    "Democratic Republic of the Congo": "Africa", "Gabon": "Africa", "Guinea": "Africa",
    "Benin": "Africa", "Burkina Faso": "Africa", "Burundi": "Africa", "Chad": "Africa",
    "Djibouti": "Africa", "Equatorial Guinea": "Africa", "Eritrea": "Africa",
    "Gambia": "Africa", "Liberia": "Africa", "Mauritania": "Africa", "Niger": "Africa",
    "Sierra Leone": "Africa", "Somalia": "Africa", "Togo": "Africa", "Libya": "Africa",

    # Indian Subcontinent
    "India": "Indian Subcontinent", "Sri Lanka": "Indian Subcontinent",
    "Bangladesh": "Indian Subcontinent", "Pakistan": "Indian Subcontinent",
    "Nepal": "Indian Subcontinent", "Maldives": "Indian Subcontinent",
    "Bhutan": "Indian Subcontinent",

    # Asia-Pacific
    "Singapore": "Asia-Pacific", "Thailand": "Asia-Pacific", "China": "Asia-Pacific",
    "Australia": "Asia-Pacific", "Malaysia": "Asia-Pacific", "Hong Kong": "Asia-Pacific",
    "Korea, Republic of": "Asia-Pacific", "South Korea": "Asia-Pacific", "Japan": "Asia-Pacific",
    "Indonesia": "Asia-Pacific", "Viet Nam": "Asia-Pacific", "Vietnam": "Asia-Pacific",
    "Philippines": "Asia-Pacific", "New Zealand": "Asia-Pacific", "Taiwan": "Asia-Pacific",
    "Taiwan, Province of China": "Asia-Pacific", "Kazakhstan": "Asia-Pacific",
    "Uzbekistan": "Asia-Pacific", "Cambodia": "Asia-Pacific", "Myanmar": "Asia-Pacific",
    "Afghanistan": "Asia-Pacific", "Brunei Darussalam": "Asia-Pacific", "Brunei": "Asia-Pacific",
    "Fiji": "Asia-Pacific", "Papua New Guinea": "Asia-Pacific", "Mongolia": "Asia-Pacific",
    "Kyrgyzstan": "Asia-Pacific", "Tajikistan": "Asia-Pacific", "Turkmenistan": "Asia-Pacific",
    "Lao People's Democratic Republic": "Asia-Pacific", "Laos": "Asia-Pacific",
    "Macao": "Asia-Pacific", "Macau": "Asia-Pacific", "Solomon Islands": "Asia-Pacific",
    "Vanuatu": "Asia-Pacific", "Samoa": "Asia-Pacific", "Tonga": "Asia-Pacific",
}


# ------------------------------------------------------------------------------
# EXACT PORT / GATEWAY LOOKUP TABLE
# ------------------------------------------------------------------------------

EXACT_PORT_MAP: Dict[str, str] = {
    # Tricky Port Disambiguations
    "NCE": "Europe",
    "NICE": "Europe",
    "NICE CÔTE D'AZUR AIRPORT": "Europe",
    "NICE COTE D'AZUR AIRPORT": "Europe",
    "AÉROPORT DE NICE": "Europe",
    "AEROPORT DE NICE": "Europe",

    "MLE": "Indian Subcontinent",
    "MALE": "Indian Subcontinent",
    "MALE INTERNATIONAL AIRPORT": "Indian Subcontinent",
    "VELANA INTERNATIONAL AIRPORT": "Indian Subcontinent",

    "YEREVAN": "Europe",
    "EVN": "Europe",
    "ZVARTNOTS": "Europe",

    "EL EDEN": "Latin America",
    "EL EDEN AIRPORT": "Latin America",
    "AXM": "Latin America",

    "CALI": "Latin America",
    "CLO": "Latin America",
    "ALFONSO BONILLA ARAGON": "Latin America",

    "YAOUNDE": "Africa",
    "NSI": "Africa",
    "NSIMALEN": "Africa",

    "YAO": "Asia-Pacific",

    # --------------------------------------------------------------------------
    # Expanded Gateways - Europe
    # --------------------------------------------------------------------------
    "SHEREMETYEVO INTERNATIONAL AIRPORT": "Europe",
    "SHEREMETYEVO": "Europe",
    "SVO": "Europe",
    "WARSZAWA": "Europe",
    "WAW": "Europe",
    "EL PRAT DE LLOBREGAT": "Europe",
    "EL PRAT": "Europe",
    "BCN": "Europe",
    "OTOPENI APT/BUCURESTI": "Europe",
    "OTOPENI": "Europe",
    "OTP": "Europe",
    "HANNOVER": "Europe",
    "HAJ": "Europe",
    "KOLN": "Europe",
    "CGN": "Europe",
    "KOLN/BONN": "Europe",
    "SAINT PETERSBURG (EX LENINGRAD)": "Europe",
    "SAINT PETERSBURG": "Europe",
    "LED": "Europe",
    "CALDEN": "Europe",
    "KSF": "Europe",
    "TBILISI": "Europe",
    "TBS": "Europe",
    "MUNSTER": "Europe",
    "BORDEAUX": "Europe",
    "BOD": "Europe",
    "SEVILLA": "Europe",
    "SVQ": "Europe",
    "DRESDEN": "Europe",
    "DRS": "Europe",
    "NUREMBERG": "Europe",
    "NURNBERG": "Europe",
    "NUE": "Europe",
    "BILBAO": "Europe",
    "BIO": "Europe",
    "BASEL": "Europe",
    "BSL": "Europe",
    "EUROAIRPORT": "Europe",
    "KATOWICE": "Europe",
    "KTW": "Europe",
    "VILNIUS": "Europe",
    "VNO": "Europe",
    "RIGA": "Europe",
    "RIX": "Europe",
    "TALLINN": "Europe",
    "TLL": "Europe",
    "BOLOGNA": "Europe",
    "BLQ": "Europe",
    "SARAJEVO": "Europe",
    "SJJ": "Europe",
    "NEWCASTLE UPON TYNE": "Europe",
    "NCL": "Europe",
    "VALENCIA": "Europe",
    "VLC": "Europe",
    "NANTES": "Europe",
    "NTE": "Europe",
    "VENEZIA (VENICE)": "Europe",
    "VENEZIA": "Europe",
    "VENICE": "Europe",
    "VCE": "Europe",
    "ANTALYA": "Europe",
    "AYT": "Europe",
    "MAASTRICHT": "Europe",
    "MST": "Europe",
    "GENOVA": "Europe",
    "GOA": "Europe",
    "LUQA": "Europe",
    "MLA": "Europe",
    "MALTA": "Europe",
    "BAKU": "Europe",
    "GYD": "Europe",
    "HEYDAR ALIYEV INTERNATIONAL APT.": "Europe",
    "HEYDAR ALIYEV": "Europe",
    "TOULOUSE": "Europe",
    "TLS": "Europe",
    "MINSK": "Europe",
    "MSQ": "Europe",
    "LARNACA": "Europe",
    "LCA": "Europe",
    "IZMIR ADNAN MENDERES INTERNATIONAL AIRPORT": "Europe",
    "IZMIR": "Europe",
    "ADB": "Europe",
    "PRISTINA": "Europe",
    "PRN": "Europe",
    "LILLE-LESQUIN APT": "Europe",
    "LILLE": "Europe",
    "LIL": "Europe",
    "ESENBOGA APT/ANKARA": "Europe",
    "ANKARA": "Europe",
    "ESB": "Europe",
    "BELFAST": "Europe",
    "BFS": "Europe",
    "BHD": "Europe",
    "MUNSTER-OSNABRUCK INTERNATIONAL APT": "Europe",
    "FMO": "Europe",
    "RUSE": "Europe",
    "ROU": "Europe",
    "GENEVE": "Europe",
    "GVA": "Europe",
    "GLASGOW": "Europe",
    "GLA": "Europe",
    "KRAKOW": "Europe",
    "KRK": "Europe",
    "COVILHA": "Europe",
    "COV": "Europe",
    "FIRENZE": "Europe",
    "FLORENCE": "Europe",
    "FLR": "Europe",
    "STAVANGER": "Europe",
    "SVG": "Europe",
    "CAEN": "Europe",
    "CFR": "Europe",
    "BUCURESTI": "Europe",
    "BUCURESTI-BANEASA APT": "Europe",
    "BBU": "Europe",
    "IRKUTSK": "Europe",
    "IKT": "Europe",
    "MOSKVA": "Europe",
    "BREMEN": "Europe",
    "BRE": "Europe",
    "GRAZ": "Europe",
    "GRZ": "Europe",
    "EDINBURGH": "Europe",
    "EDI": "Europe",
    "NAPOLI": "Europe",
    "NAPLES": "Europe",
    "NAP": "Europe",
    "NOVOSIBIRSK": "Europe",
    "OVB": "Europe",
    "MALMO": "Europe",
    "MALMO AIRPORT": "Europe",
    "MMX": "Europe",
    "LAS PALMAS DE GRAN CANARIA": "Europe",
    "LPA": "Europe",
    "CORK": "Europe",
    "ORK": "Europe",
    "LJUBLJANA": "Europe",
    "LJU": "Europe",
    "PLOVDIV": "Europe",
    "PDV": "Europe",
    "BILLUND": "Europe",
    "BLL": "Europe",
    "ABERDEEN": "Europe",
    "ABZ": "Europe",
    "LEIPZIG": "Europe",
    "LEIPZIG/HALLE": "Europe",
    "LEJ": "Europe",
    "MULHOUSE": "Europe",
    "MLH": "Europe",
    "STRASBOURG": "Europe",
    "SXB": "Europe",
    "ALICANTE": "Europe",
    "ALC": "Europe",
    "THESSALONIKI": "Europe",
    "SKG": "Europe",
    "NOTODDEN": "Europe",
    "CAGLIARI": "Europe",
    "CAG": "Europe",
    "LINZ": "Europe",
    "LNZ": "Europe",
    "TENERIFE": "Europe",
    "TFN": "Europe",
    "TFS": "Europe",
    "TAMPERE (TAMMERFORS)": "Europe",
    "TAMPERE": "Europe",
    "TMP": "Europe",
    "SKOPJE": "Europe",
    "SKP": "Europe",
    "GDANSK": "Europe",
    "GDN": "Europe",
    "LIEGE": "Europe",
    "LGG": "Europe",
    "ADANA": "Europe",
    "ADA": "Europe",
    "ERFURT": "Europe",
    "ERF": "Europe",

    # --------------------------------------------------------------------------
    # Expanded Gateways - Africa
    # --------------------------------------------------------------------------
    "SIR SEEWOOSAGUR RAMGOOLAM INT APT": "Africa",
    "SIR SEEWOOSAGUR RAMGOOLAM": "Africa",
    "MRU": "Africa",
    "MOHAMMED V INTERNATIONAL AIRPORT": "Africa",
    "MOHAMMED V": "Africa",
    "CMN": "Africa",
    "LOME": "Africa",
    "LFW": "Africa",
    "BLAISE DIAGNE INTERNATIONAL AIRPORT": "Africa",
    "BLAISE DIAGNE": "Africa",
    "DSS": "Africa",
    "KINSHASA": "Africa",
    "FIH": "Africa",
    "ANTANANARIVO": "Africa",
    "TNR": "Africa",
    "MAHE ISLAND APT": "Africa",
    "MAHE ISLAND": "Africa",
    "SEZ": "Africa",
    "OUAGADOUGOU": "Africa",
    "OUA": "Africa",
    "DOUALA": "Africa",
    "DLA": "Africa",
    "KIGALI": "Africa",
    "KGL": "Africa",
    "WINDHOEK": "Africa",
    "WDH": "Africa",
    "BUJUMBURA": "Africa",
    "BJM": "Africa",
    "LIBREVILLE": "Africa",
    "LBV": "Africa",
    "HARGEISA": "Africa",
    "HGA": "Africa",
    "BRAZZAVILLE": "Africa",
    "BZV": "Africa",
    "KILIMANJARO": "Africa",
    "JRO": "Africa",
    "DJIBOUTI": "Africa",
    "JIB": "Africa",
    "COTONOU": "Africa",
    "COO": "Africa",
    "FREETOWN": "Africa",
    "FNA": "Africa",
    "ZANZIBAR": "Africa",
    "ZNZ": "Africa",
    "ROBERTS INTERNATIONAL AIRPORT": "Africa",
    "ROBERTS": "Africa",
    "MONROVIA": "Africa",
    "ROB": "Africa",
    "LILONGWE": "Africa",
    "LLW": "Africa",
    "CONAKRY": "Africa",
    "CKY": "Africa",
    "N'DJAMENA": "Africa",
    "NDJAMENA": "Africa",
    "NDJ": "Africa",
    "BANJUL": "Africa",
    "BJL": "Africa",
    "BLANTYRE": "Africa",
    "BLZ": "Africa",
    "JUBA": "Africa",
    "JUB": "Africa",
    "ASMARA": "Africa",
    "ASM": "Africa",
    "POINTE NOIRE": "Africa",
    "PNR": "Africa",
    "NIAMEY": "Africa",
    "NIM": "Africa",
    "BAMAKO": "Africa",
    "BKO": "Africa",
    "KANO": "Africa",
    "KAN": "Africa",
    "PORT HARCOURT": "Africa",
    "PHC": "Africa",
    "ABUJA": "Africa",
    "ABV": "Africa",
    "BENGUELA": "Africa",
    "BUG": "Africa",
    "GABORONE": "Africa",
    "GBE": "Africa",
    "ENFIDHA HAMMAMET INTL APT": "Africa",
    "ENFIDHA": "Africa",
    "NBE": "Africa",
    "MOMBASA": "Africa",
    "MBA": "Africa",
    "PORT GENTIL": "Africa",
    "POG": "Africa",
    "NOUAKCHOTT": "Africa",
    "NKC": "Africa",
    "LUBUMBASHI": "Africa",
    "FBM": "Africa",
    "MALABO": "Africa",
    "SSG": "Africa",
    "SIERRA LEONE APT": "Africa",
    "PORT SUDAN": "Africa",
    "PZU": "Africa",
    "DR. ANTONIO AGOSTINHO NETO INTERNATIONAL AIRPORT": "Africa",
    "AGOSTINHO NETO": "Africa",
    "MOGADISHU": "Africa",
    "MGQ": "Africa",
    "ABOU-DEIA": "Africa",
    "PORT ELIZABETH": "Africa",
    "PLZ": "Africa",
    "ALEXANDRIA": "Africa",
    "HBE": "Africa",
    "NOUADHIBOU": "Africa",
    "NDB": "Africa",

    # --------------------------------------------------------------------------
    # Expanded Gateways - Asia-Pacific
    # --------------------------------------------------------------------------
    "KANSAI INT APT": "Asia-Pacific",
    "KANSAI": "Asia-Pacific",
    "KIX": "Asia-Pacific",
    "TASHKENT": "Asia-Pacific",
    "TAS": "Asia-Pacific",
    "ALMATY": "Asia-Pacific",
    "ALA": "Asia-Pacific",
    "YANGON": "Asia-Pacific",
    "RGN": "Asia-Pacific",
    "VIENTIANE": "Asia-Pacific",
    "VTE": "Asia-Pacific",
    "PHNOM PENH": "Asia-Pacific",
    "PNH": "Asia-Pacific",
    "ULAANBAATAR": "Asia-Pacific",
    "ULN": "Asia-Pacific",
    "DUSHANBE": "Asia-Pacific",
    "DYU": "Asia-Pacific",
    "BISHKEK": "Asia-Pacific",
    "FRU": "Asia-Pacific",
    "ASHGABAT": "Asia-Pacific",
    "ASB": "Asia-Pacific",
    "FUKUOKA": "Asia-Pacific",
    "FUK": "Asia-Pacific",
    "SAPPORO": "Asia-Pacific",
    "CTS": "Asia-Pacific",
    "CEBU": "Asia-Pacific",
    "CEB": "Asia-Pacific",
    "QINGDAO LIUTING INTERNATIONAL APT": "Asia-Pacific",
    "QINGDAO": "Asia-Pacific",
    "TAO": "Asia-Pacific",
    "DALIAN ZHOUSHUIZI INTERNATIONAL APT": "Asia-Pacific",
    "DALIAN": "Asia-Pacific",
    "DLC": "Asia-Pacific",
    "NAGOYA, AICHI": "Asia-Pacific",
    "NAGOYA": "Asia-Pacific",
    "NGO": "Asia-Pacific",
    "CHENGDU TIANFU INTERNATIONAL AIRPORT": "Asia-Pacific",
    "CHENGDU SHUANGLIU INTERNATIONAL APT": "Asia-Pacific",
    "CHENGDU": "Asia-Pacific",
    "TFU": "Asia-Pacific",
    "CTU": "Asia-Pacific",
    "PENANG (GEORGETOWN)": "Asia-Pacific",
    "PENANG": "Asia-Pacific",
    "PEN": "Asia-Pacific",
    "SURABAYA": "Asia-Pacific",
    "SUB": "Asia-Pacific",
    "WUHAN TIANHE INTERNATIONAL APT": "Asia-Pacific",
    "WUHAN": "Asia-Pacific",
    "WUH": "Asia-Pacific",
    "KAOHSIUNG": "Asia-Pacific",
    "KHH": "Asia-Pacific",
    "NURSULTAN NAZARBAYEV INTERNATIONAL AIRPORT": "Asia-Pacific",
    "ASTANA": "Asia-Pacific",
    "NQZ": "Asia-Pacific",
    "TSE": "Asia-Pacific",
    "SHENZHEN BAOAN INTERNATIONAL APT": "Asia-Pacific",
    "SHENZHEN": "Asia-Pacific",
    "SZX": "Asia-Pacific",
    "NINGBO LISHE INTERNATIONAL APT": "Asia-Pacific",
    "NINGBO": "Asia-Pacific",
    "NGB": "Asia-Pacific",
    "BUSAN": "Asia-Pacific",
    "PUS": "Asia-Pacific",
    "NUKU'ALOFA": "Asia-Pacific",
    "TBU": "Asia-Pacific",
    "BANDAR SERI BEGAWAN": "Asia-Pacific",
    "BWN": "Asia-Pacific",
    "TAN SON NHAT APT": "Asia-Pacific",
    "SGN": "Asia-Pacific",
    "NADI": "Asia-Pacific",
    "NAN": "Asia-Pacific",
    "ASHKHABAD": "Asia-Pacific",
    "LABUAN, SABAH": "Asia-Pacific",
    "LBU": "Asia-Pacific",
    "DA NANG": "Asia-Pacific",
    "DAD": "Asia-Pacific",
    "TAPAKTUAN, SUMATRA": "Asia-Pacific",
    "BISHKEK (EX FRUNZE)": "Asia-Pacific",
    "ATYRAU (EX GURYEV)": "Asia-Pacific",
    "ATYRAU": "Asia-Pacific",
    "GUW": "Asia-Pacific",
    "HANGZHOU XIAOSHAN INTERNATIONAL APT": "Asia-Pacific",
    "HANGZHOU": "Asia-Pacific",
    "HGH": "Asia-Pacific",
    "CHRISTCHURCH": "Asia-Pacific",
    "CHC": "Asia-Pacific",
    "OSAKA": "Asia-Pacific",
    "OSA": "Asia-Pacific",
    "ITM": "Asia-Pacific",
    "PAPEETE": "Asia-Pacific",
    "PPT": "Asia-Pacific",
    "QIQIHAR SANJIAZI APT": "Asia-Pacific",
    "NDG": "Asia-Pacific",
    "ULANHAD TUCHENGCITY APT": "Asia-Pacific",
    "NANJING LUKOU INTERNATIONAL APT": "Asia-Pacific",
    "NANJING": "Asia-Pacific",
    "NKG": "Asia-Pacific",
    "CHIANG MAI INTERNATIONAL APT": "Asia-Pacific",
    "CHIANG MAI": "Asia-Pacific",
    "CNX": "Asia-Pacific",
    "TECHO INTERNATIONAL AIRPORT": "Asia-Pacific",
    "KTI": "Asia-Pacific",
    "DONGHAI AIRLINES": "Asia-Pacific",
    "ZHENGZHOU XINZHENG INTERNATIONAL APT": "Asia-Pacific",
    "ZHENGZHOU": "Asia-Pacific",
    "CGO": "Asia-Pacific",
    "SAMARKAND": "Asia-Pacific",
    "SKD": "Asia-Pacific",
    "AKTAU APT": "Asia-Pacific",
    "AKTAU": "Asia-Pacific",
    "SCO": "Asia-Pacific",
    "MANDALAY": "Asia-Pacific",
    "MDL": "Asia-Pacific",
    "PORT MORESBY": "Asia-Pacific",
    "POM": "Asia-Pacific",
    "SUVA": "Asia-Pacific",
    "SUV": "Asia-Pacific",
    "MEDAN, SUMATRA": "Asia-Pacific",
    "MEDAN": "Asia-Pacific",
    "KNO": "Asia-Pacific",
    "GOULBURN ISLAND": "Asia-Pacific",
    "CLARK AIRPORT": "Asia-Pacific",
    "CRK": "Asia-Pacific",
    "CHONGQING JIANGBEI INTERNATIONAL APT": "Asia-Pacific",
    "CHONGQING": "Asia-Pacific",
    "CKG": "Asia-Pacific",
    "TAIYUAN WUSU INTERNATIONAL APT": "Asia-Pacific",
    "TAIYUAN": "Asia-Pacific",
    "TYN": "Asia-Pacific",
    "HIROSHIMA": "Asia-Pacific",
    "HIJ": "Asia-Pacific",
    "CHARLEVILLE": "Asia-Pacific",
    "CTL": "Asia-Pacific",

    # --------------------------------------------------------------------------
    # Expanded Gateways - Middle East
    # --------------------------------------------------------------------------
    "BEN GURION INTERNATIONAL APT": "Middle East",
    "BEN GURION": "Middle East",
    "TLV": "Middle East",
    "GASSIM": "Middle East",
    "ELQ": "Middle East",
    "SULAYMANIYAH INTERNATIONAL AIRPORT": "Middle East",
    "ISU": "Middle East",
    "BASRA": "Middle East",
    "BSR": "Middle East",
    "TALLIL": "Middle East",
    "DAMASCUS (DAMAS)": "Middle East",
    "DAMASCUS": "Middle East",
    "DAM": "Middle East",
    "NAJAF": "Middle East",
    "AL NAJAF": "Middle East",
    "NJF": "Middle East",
    "ABHA": "Middle East",
    "AHB": "Middle East",
    "TABUK": "Middle East",
    "TUU": "Middle East",
    "YANBU": "Middle East",
    "YNB": "Middle East",
    "TAIF": "Middle East",
    "TIF": "Middle East",
    "HAMADAN": "Middle East",
    "HDX": "Middle East",

    # --------------------------------------------------------------------------
    # Expanded Gateways - Indian Subcontinent
    # --------------------------------------------------------------------------
    "KABUL": "Indian Subcontinent",
    "KBL": "Indian Subcontinent",
    "BANGALORE": "Indian Subcontinent",
    "BENGALURU": "Indian Subcontinent",
    "BLR": "Indian Subcontinent",
    "INDIRA GANDHI INTERNATIONAL AIRPORT": "Indian Subcontinent",
    "DELHI": "Indian Subcontinent",
    "NEW DELHI": "Indian Subcontinent",
    "DEL": "Indian Subcontinent",
    "CHHATRAPATI SHIVAJI MAHARAJ INTERNATIONAL AIRPORT": "Indian Subcontinent",
    "MUMBAI": "Indian Subcontinent",
    "BOM": "Indian Subcontinent",
    "CHENNAI": "Indian Subcontinent",
    "MADRAS": "Indian Subcontinent",
    "MAA": "Indian Subcontinent",
    "GOA": "Indian Subcontinent",
    "GOI": "Indian Subcontinent",
    "GOX": "Indian Subcontinent",
    "HYDERABAD": "Indian Subcontinent",
    "HYD": "Indian Subcontinent",
    "KOLKATA": "Indian Subcontinent",
    "CALCUTTA": "Indian Subcontinent",
    "CCU": "Indian Subcontinent",
    "COCHIN": "Indian Subcontinent",
    "KOCHI": "Indian Subcontinent",
    "COK": "Indian Subcontinent",
    "PESHAWAR": "Indian Subcontinent",
    "PEW": "Indian Subcontinent",
    "PARO": "Indian Subcontinent",
    "PBH": "Indian Subcontinent",
    "AHMEDABAD": "Indian Subcontinent",
    "AMD": "Indian Subcontinent",
    "SIALKOT": "Indian Subcontinent",
    "SKT": "Indian Subcontinent",
    "MULTAN": "Indian Subcontinent",
    "MUX": "Indian Subcontinent",
    "FAISALABAD": "Indian Subcontinent",
    "LYP": "Indian Subcontinent",
    "JAMNAGAR": "Indian Subcontinent",
    "JGA": "Indian Subcontinent",
    "COIMBATORE": "Indian Subcontinent",
    "CJB": "Indian Subcontinent",
    "KUSHOK BAKULA RIMPOCHEE AIRPORT - LEH": "Indian Subcontinent",
    "LEH": "Indian Subcontinent",
    "IXL": "Indian Subcontinent",
    "DARCHULA": "Indian Subcontinent",
    "BELGAUM": "Indian Subcontinent",
    "IXG": "Indian Subcontinent",

    # --------------------------------------------------------------------------
    # Expanded Gateways - Latin America
    # --------------------------------------------------------------------------
    "LA AURORA INTERNATIONAL AIRPORT": "Latin America",
    "LA AURORA": "Latin America",
    "GUA": "Latin America",
    "KINGSTON": "Latin America",
    "KIN": "Latin America",
    "GRAND CAYMAN": "Latin America",
    "GCM": "Latin America",
    "MONTEGO BAY": "Latin America",
    "MBJ": "Latin America",
    "SAN PEDRO SULA": "Latin America",
    "SAP": "Latin America",
    "SAPIRANGA": "Latin America",
    "PIARCO INTERNATIONAL AIRPORT": "Latin America",
    "POS": "Latin America",
    "PORT OF SPAIN": "Latin America",
    "TEGUCIGALPA": "Latin America",
    "TGU": "Latin America",
    "SAINT KITTS": "Latin America",
    "SKB": "Latin America",
    "HEWANORRA INTL  APT": "Latin America",
    "HEWANORRA INTL APT": "Latin America",
    "UVF": "Latin America",
    "VIRU VIRU": "Latin America",
    "VVI": "Latin America",
    "PARAMARIBO": "Latin America",
    "PBM": "Latin America",
    "BAHIA BLANCA": "Latin America",
    "BHI": "Latin America",
    "CASIGUA": "Latin America",
    "PORT-AU-PRINCE": "Latin America",
    "PAP": "Latin America",
    "FELIPE ANGELES INTERNATIONAL AIRPORT": "Latin America",
    "NLU": "Latin America",
    "LA PAZ": "Latin America",
    "LPB": "Latin America",
    "MANAGUA": "Latin America",
    "MGA": "Latin America",
    "MONTERIA": "Latin America",
    "MTR": "Latin America",
    "TUCUPITA": "Latin America",
    "ST. MARTIN (SXM) AIRPORT": "Latin America",
    "SXM": "Latin America",
    "GEORGETOWN": "Latin America",
    "GEO": "Latin America",
    "CURACAO": "Latin America",
    "CURAÇAO": "Latin America",
    "CUR": "Latin America",
    "SAN JUAN": "Latin America",
    "SJU": "Latin America",
    "PUNTA CANA": "Latin America",
    "PUJ": "Latin America",
    "BONAIRE": "Latin America",
    "BON": "Latin America",
    "BARRANQUILLA": "Latin America",
    "BAQ": "Latin America",
    "BELO HORIZONTE": "Latin America",
    "CNF": "Latin America",
    "BRASILIA": "Latin America",
    "BSB": "Latin America",
    "BRIDGETOWN": "Latin America",
    "BGI": "Latin America",
    "CONFINS INTERNATIONAL": "Latin America",
    "CONFINS": "Latin America",
    "LA HABANA": "Latin America",
    "HAVANA": "Latin America",
    "HAV": "Latin America",
    "TOBAGO": "Latin America",
    "TAB": "Latin America",
    "NASSAU": "Latin America",
    "NAS": "Latin America",

    # --------------------------------------------------------------------------
    # Expanded Gateways - North America
    # --------------------------------------------------------------------------
    "KANSAS CITY": "North America",
    "MCI": "North America",
    "AUSTIN": "North America",
    "AUS": "North America",
    "NASHVILLE": "North America",
    "BNA": "North America",
    "INDIANAPOLIS": "North America",
    "IND": "North America",
    "RALEIGH": "North America",
    "RDU": "North America",
    "CLEVELAND": "North America",
    "CLE": "North America",
    "CINCINNATI": "North America",
    "CVG": "North America",
    "SALT LAKE CITY": "North America",
    "SLC": "North America",
    "MINNEAPOLIS": "North America",
    "MSP": "North America",
    "MISSION-MCALLEN APT": "North America",
    "MCALLEN": "North America",
    "MFE": "North America",
    "WINNIPEG": "North America",
    "YWG": "North America",
    "TAMPA": "North America",
    "TPA": "North America",
    "NEW ORLEANS": "North America",
    "MSY": "North America",
    "LAREDO": "North America",
    "LRD": "North America",
    "PITTSBURGH": "North America",
    "PITTSBURG": "North America",
    "PIT": "North America",
    "EDMONTON INT APT": "North America",
    "EDMONTON": "North America",
    "YEG": "North America",
    "EL PASO": "North America",
    "ELP": "North America",
    "MONROE": "North America",
    "MLU": "North America",
    "SAINT LOUIS": "North America",
    "ST. LOUIS": "North America",
    "STL": "North America",
    "NORFOLK": "North America",
    "ORF": "North America",
    "DES MOINES": "North America",
    "DSM": "North America",
    "BUFFALO": "North America",
    "BUF": "North America",
    "CHARLESTON": "North America",
    "CHS": "North America",
    "FORT LAUDERDALE": "North America",
    "FLL": "North America",
    "KNOXVILLE": "North America",
    "TYS": "North America",
    "RALEIGH-DURHAM APT": "North America",
    "SAN JOSE": "North America",
    "SJC": "North America",
    "LAS VEGAS": "North America",
    "LAS": "North America",
    "PORTLAND": "North America",
    "PDX": "North America",
    "SAN ANTONIO": "North America",
    "SAT": "North America",
    "JACKSONVILLE": "North America",
    "JAX": "North America",
    "BALTIMORE": "North America",
    "BWI": "North America",
    "RICHMOND": "North America",
    "RIC": "North America",
    "HARTFORD": "North America",
    "BDL": "North America",
    "COLUMBUS": "North America",
    "CMH": "North America",
    "MEMPHIS": "North America",
    "MEM": "North America",
    "LOUISVILLE": "North America",
    "SDF": "North America",
    "MILWAUKEE": "North America",
    "MKE": "North America",
    "SACRAMENTO": "North America",
    "SMF": "North America",
    "HALIFAX": "North America",
    "YHZ": "North America",
    "ANAHEIM": "North America",
    "CLAYTON": "North America",
    "SIOUX FALLS": "North America",
    "FSD": "North America",
    "MOUNT CLEMENS": "North America",
    "SAN DIEGO": "North America",
    "SAN": "North America",
    "COLORADO SPRINGS": "North America",
    "COS": "North America",
    "OMAHA": "North America",
    "OMA": "North America",
    "NEWBURGH": "North America",
    "SWF": "North America",
    "BINGHAMTON": "North America",
    "BGM": "North America",
    "DAYTON": "North America",
    "DAY": "North America",
}


# ------------------------------------------------------------------------------
# REGIONAL WORD-BOUNDARY PATTERNS (KEYWORD FALLBACK)
# ------------------------------------------------------------------------------

REGION_PATTERNS = [
    ("Indian Subcontinent", re.compile(
        r"\b(?:"
        r"MALE|COLOMBO|DHAKA|KARACHI|LAHORE|KATHMANDU|ISLAMABAD|CHITTAGONG|"
        r"MLE|CMB|DAC|KHI|LHE|KTM|ISB|BANDARANAIKE|HAZRAT\s+SHAHJALAL|"
        r"BANGALORE|BENGALURU|BLR|DELHI|NEW\s+DELHI|DEL|MUMBAI|BOM|CHENNAI|MADRAS|MAA|"
        r"GOA|GOI|GOX|HYDERABAD|HYD|KOLKATA|CALCUTTA|CCU|COCHIN|KOCHI|COK|"
        r"PESHAWAR|PEW|PARO|PBH|AHMEDABAD|AMD|SIALKOT|SKT|MULTAN|MUX|FAISALABAD|LYP|"
        r"JAMNAGAR|COIMBATORE|LEH|DARCHULA|BELGAUM"
        r")\b",
        re.IGNORECASE,
    )),
    ("Middle East", re.compile(
        r"\b(?:"
        r"DUBAI|ZAYED|ABU\s+DHABI|SHARJAH|MUSCAT|DOHA|DAMMAM|JEDDAH|RIYADH|BAHRAIN|KUWAIT|"
        r"SALALAH|AMMAN|BEIRUT|BAGHDAD|ERBIL|RAS\s+AL\s+KHAIMAH|TEL\s+AVIV|MEDINA|MADINAH|TEHRAN|"
        r"DXB|AUH|DOH|JED|RUH|BEN\s+GURION|TLV|GASSIM|ELQ|SULAYMANIYAH|ISU|BASRA|BSR|TALLIL|"
        r"DAMASCUS|DAM|NAJAF|NJF|ABHA|TABUK|YANBU|TAIF"
        r")\b",
        re.IGNORECASE,
    )),
    ("Europe", re.compile(
        r"\b(?:"
        r"HEATHROW|LONDON|FRANKFURT|AMSTERDAM|PARIS|CHARLES\s+DE\s+GAULLE|MILAN|MALPENSA|ROME|"
        r"MADRID|BARCELONA|ZURICH|DUBLIN|BRUSSELS|BRUXELLES|VIENNA|MUNICH|MANCHESTER|BIRMINGHAM|"
        r"ISTANBUL|ATHENS|WARSAW|PRAGUE|PRAHA|LISBON|LISBOA|BUDAPEST|STOCKHOLM|COPENHAGEN|OSLO|"
        r"HAMBURG|WIEN|MOSCOW|STUTTGART|GENEVA|HELSINKI|DUSSELDORF|PORTO|FIUMICINO|BELGRADE|"
        r"BRATISLAVA|NICE|GOTHENBURG|VILNIUS|MARSEILLE|SOFIA|MALAGA|ZAGREB|BERLIN|LYON|ROTTERDAM|"
        r"LUXEMBOURG|LHR|FRA|CDG|AMS|MAD|BCN|MUC|FCO|IST|"
        r"SHEREMETYEVO|WARSZAWA|OTOPENI|HANNOVER|KOLN|BONN|TBILISI|BORDEAUX|SEVILLA|DRESDEN|"
        r"NUREMBERG|NURNBERG|BILBAO|BASEL|KATOWICE|RIGA|TALLINN|CALDEN|BOLOGNA|SARAJEVO|VALENCIA|"
        r"NANTES|VENICE|VENEZIA|ANTALYA|MAASTRICHT|GENOVA|LUQA|MALTA|BAKU|TOULOUSE|MINSK|LARNACA|"
        r"IZMIR|PRISTINA|LILLE|ANKARA|BELFAST|GLASGOW|KRAKOW|FLORENCE|FIRENZE|STAVANGER|BREMEN|"
        r"GRAZ|EDINBURGH|NAPLES|NAPOLI|NOVOSIBIRSK|MALMO|BILLUND|ABERDEEN|LEIPZIG|STRASBOURG|"
        r"ALICANTE|THESSALONIKI|CAGLIARI|LINZ|TENERIFE|SVO|WAW|OTP|HAJ|CGN|LED|KSF|TBS"
        r")\b",
        re.IGNORECASE,
    )),
    ("North America", re.compile(
        r"\b(?:"
        r"KENNEDY|NEW\s+YORK|JFK|PEARSON|TORONTO|MONTREAL|VANCOUVER|CHICAGO|LOS\s+ANGELES|LAX|"
        r"ATLANTA|DALLAS|HOUSTON|MIAMI|SAN\s+FRANCISCO|BOSTON|SEATTLE|DETROIT|NEWARK|PHILADELPHIA|"
        r"CALGARY|OTTAWA|DORVAL|CHARLOTTE|DULLES|WASHINGTON|DENVER|ORLANDO|COLUMBUS|PHOENIX|"
        r"MEMPHIS|ORD|DFW|SFO|MIA|EWR|BOS|COLUMBIA|"
        r"KANSAS\s+CITY|AUSTIN|NASHVILLE|INDIANAPOLIS|RALEIGH|CLEVELAND|CINCINNATI|SALT\s+LAKE\s+CITY|"
        r"MINNEAPOLIS|MCALLEN|WINNIPEG|TAMPA|NEW\s+ORLEANS|LAREDO|PITTSBURGH|EDMONTON|EL\s+PASO|"
        r"MONROE|SAINT\s+LOUIS|ST\.\s+LOUIS|NORFOLK|DES\s+MOINES|BUFFALO|CHARLESTON|FORT\s+LAUDERDALE|"
        r"KNOXVILLE|LAS\s+VEGAS|PORTLAND|SAN\s+ANTONIO|JACKSONVILLE|BALTIMORE|RICHMOND|HARTFORD|"
        r"LOUISVILLE|MILWAUKEE|SACRAMENTO|HALIFAX|ANAHEIM|SAN\s+DIEGO|COLORADO\s+SPRINGS|OMAHA|"
        r"MCI|AUS|BNA|RDU|CLE|CVG|SLC|MSP|TPA|MSY|LRD|PIT|ELP|BUF|FLL|LAS|PDX|SAT|JAX|BWI|RIC|BDL|CMH|MEM|SDF|MKE|SMF|SAN|COS|OMA"
        r")\b",
        re.IGNORECASE,
    )),
    ("Latin America", re.compile(
        r"\b(?:"
        r"SAO\s+PAULO|BUENOS\s+AIRES|MEXICO(?:\s+CITY)?|BOGOTA|SANTIAGO|LIMA|PANAMA(?:\s+CITY)?|"
        r"RIO(?:\s+DE\s+JANEIRO)?|QUITO|SANTO\s+DOMINGO|CARACAS|GUAYAQUIL|ASUNCION|MONTERREY|"
        r"VIRACOPOS|CURITIBA|SAN\s+SALVADOR|MONTEVIDEO|GUADALAJARA|CALI|MEDELLIN|CANCUN|"
        r"CARTAGEN(?:A)?|GRU|GIG|BOG|EZE|SCL|LIM|PTY|EL\s+EDEN|"
        r"LA\s+AURORA|KINGSTON|GRAND\s+CAYMAN|MONTEGO\s+BAY|SAN\s+PEDRO\s+SULA|SAPIRANGA|PIARCO|"
        r"TEGUCIGALPA|SAINT\s+KITTS|HEWANORRA|VIRU\s+VIRU|PARAMARIBO|BAHIA\s+BLANCA|CASIGUA|"
        r"PORT-AU-PRINCE|FELIPE\s+ANGELES|LA\s+PAZ|MANAGUA|MONTERIA|TUCUPITA|CURACAO|CURAÇAO|"
        r"SAN\s+JUAN|PUNTA\s+CANA|BONAIRE|PORT\s+OF\s+SPAIN|BARRANQUILLA|BELO\s+HORIZONTE|"
        r"BRASILIA|BRIDGETOWN|CONFINS|HAVANA|LA\s+HABANA|TOBAGO|GUA|KIN|GCM|MBJ|SAP|CUR|SJU|PUJ|POS|BAQ|BGI"
        r")\b",
        re.IGNORECASE,
    )),
    ("Africa", re.compile(
        r"\b(?:"
        r"CAIRO|LAGOS|NAIROBI|KENYATTA|JOHANNESBURG|OR\s+TAMBO|ADDIS\s+ABABA|ACCRA|ENTEBBE|"
        r"DAR\s+ES\s+SALAAM|KIGALI|CASABLANCA|HARARE|LUSAKA|ALGIERS|TUNIS|ABIDJAN|DAKAR|"
        r"MAPUTO|LUANDA|MAURITIUS|DURBAN|CAPE\s+TOWN|YAOUNDE|JNB|CAI|NBO|LOS|ADD|CPT|BERENICE|"
        r"SIR\s+SEEWOOSAGUR|RAMGOOLAM|MOHAMMED\s+V|LOME|BLAISE\s+DIAGNE|KINSHASA|ANTANANARIVO|"
        r"MAHE\s+ISLAND|OUAGADOUGOU|DOUALA|WINDHOEK|BUJUMBURA|LIBREVILLE|HARGEISA|BRAZZAVILLE|"
        r"KILIMANJARO|DJIBOUTI|COTONOU|FREETOWN|ZANZIBAR|ROBERTS|MONROVIA|LILONGWE|CONAKRY|"
        r"N'DJAMENA|BANJUL|BLANTYRE|JUBA|ASMARA|POINTE\s+NOIRE|NIAMEY|BAMAKO|KANO|PORT\s+HARCOURT|"
        r"ABUJA|BENGUELA|GABORONE|ENFIDHA|MOMBASA|PORT\s+GENTIL|NOUAKCHOTT|LUBUMBASHI|MALABO|"
        r"PORT\s+SUDAN|AGOSTINHO\s+NETO|MOGADISHU|BEIRA|NAMPULA|WALVIS\s+BAY|GOMA|PORT\s+ELIZABETH|"
        r"MRU|CMN|LFW|DSS|FIH|TNR|SEZ|OUA|DLA|KGL|WDH|BJM|LBV|HGA|BZV|JRO|COO|FNA|ZNZ|ROB|LLW|CKY|BJL|BLZ|JUB|ASM|NIM|BKO|KAN|PHC|ABV|MBA|PZU|MGQ|PLZ"
        r")\b",
        re.IGNORECASE,
    )),
    ("Asia-Pacific", re.compile(
        r"\b(?:"
        r"SINGAPORE|SHANGHAI|PUDONG|BANGKOK|SUVARNABHUMI|HONG\s+KONG|TOKYO|NARITA|HANEDA|SEOUL|"
        r"INCHEON|KUALA\s+LUMPUR|JAKARTA|MANILA|SYDNEY|MELBOURNE|HANOI|HO\s+CHI\s+MINH|TAIPEI|"
        r"GUANGZHOU|BEIJING|AUCKLAND|PERTH|BRISBANE|SIN|PVG|BKK|HKG|NRT|HND|ICN|KUL|CGK|MNL|SYD|MEL|YAO|"
        r"KANSAI|TASHKENT|ALMATY|YANGON|VIENTIANE|PHNOM\s+PENH|ULAANBAATAR|DUSHANBE|BISHKEK|"
        r"ASHGABAT|ASHKHABAD|FUKUOKA|SAPPORO|CEBU|QINGDAO|DALIAN|NAGOYA|CHENGDU|PENANG|SURABAYA|"
        r"WUHAN|KAOHSIUNG|NURSULTAN|ASTANA|SHENZHEN|NINGBO|BUSAN|NUKU'ALOFA|BANDAR\s+SERI\s+BEGAWAN|"
        r"TAN\s+SON\s+NHAT|NADI|LABUAN|DA\s+NANG|TAPAKTUAN|ATYRAU|HANGZHOU|CHRISTCHURCH|OSAKA|"
        r"PAPEETE|QIQIHAR|ULANHAD|NANJING|CHIANG\s+MAI|TECHO|DONGHAI|ZHENGZHOU|MACAU|MACAO|"
        r"KOSRAE|SAIPAN|GUAM|WELLINGTON|ADELAIDE|DARWIN|CAIRNS|XIAMEN|TIANJIN|SAMARKAND|AKTAU|"
        r"MANDALAY|PORT\s+MORESBY|SUVA|MEDAN|CLARK|CHONGQING|"
        r"KIX|TAS|ALA|RGN|VTE|PNH|ULN|DYU|FRU|ASB|FUK|CTS|CEB|TAO|DLC|NGO|TFU|CTU|PEN|SUB|WUH|KHH|NQZ|SZX|NGB|PUS|SGN|NAN|DAD|HGH|CHC|OSA|ITM|PPT|NKG|CNX|CGO|MFM|XMN|TSN|SKD|SCO|MDL|POM|SUV|CRK|CKG"
        r")\b",
        re.IGNORECASE,
    )),
]


def map_destination_region(port_name: str, country_name: str = "") -> str:
    """
    Map destination airport / port name and optional country name to canonical
    freight destination regions.

    Uses country-first resolution to prevent short substring collision (fixing Issue C2).
    Standard regions:
      - Europe, Middle East, North America, Asia-Pacific, Latin America, Africa, Indian Subcontinent.
      - Fallback: 'Other Destination'.
    """
    country_clean = str(country_name).strip() if country_name else ""
    port_clean = str(port_name).strip() if port_name else ""

    # 1. Country-First Resolution
    if country_clean:
        if country_clean in COUNTRY_TO_REGION:
            return COUNTRY_TO_REGION[country_clean]
        # Case-insensitive country lookup
        c_upper = country_clean.upper()
        for c_known, region in COUNTRY_TO_REGION.items():
            if c_upper == c_known.upper():
                return region

    if not port_clean:
        return "Other Destination"

    port_upper = port_clean.upper()

    # 2. Port string containing explicit country reference after comma (e.g. 'Armenia, Colombia')
    if "," in port_clean:
        sub_country = port_clean.split(",")[-1].strip()
        if sub_country in COUNTRY_TO_REGION:
            return COUNTRY_TO_REGION[sub_country]
        sub_upper = sub_country.upper()
        for c_known, region in COUNTRY_TO_REGION.items():
            if sub_upper == c_known.upper():
                return region

    # 3. Disambiguation for 'Armenia' with Colombia vs Europe context
    if "ARMENIA" in port_upper and "COLOMBIA" in port_upper:
        return "Latin America"

    # 4. Exact Port / Code Lookup
    if port_upper in EXACT_PORT_MAP:
        return EXACT_PORT_MAP[port_upper]

    for p_known, region in EXACT_PORT_MAP.items():
        if port_upper == p_known.upper():
            return region

    # 5. Check if port matches a country name directly (e.g. 'Armenia' -> Europe)
    if port_clean in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[port_clean]
    for c_known, region in COUNTRY_TO_REGION.items():
        if port_upper == c_known.upper():
            return region

    # 6. Regional Word-Boundary Patterns
    for region_name, pattern in REGION_PATTERNS:
        if pattern.search(port_clean):
            return region_name

    return "Other Destination"
