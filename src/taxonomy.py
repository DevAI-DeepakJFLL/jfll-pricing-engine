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
RE_PHARMACEUTICALS = re.compile(
    r"\b(?:"
    r"PHARMA|PHARMACEUTICALS?|MEDICINES?|MEDICATIONS?|DRUGS?|"
    r"VACCINES?|INSULIN|SERUM|BIOLOGICALS?|DIAGNOSTIC\s+KITS?|"
    r"HEALTHCARE|MEDICAL\s+DEVICES?|CLINICAL\s+TRIALS?|ANTIBIOTICS?|"
    r"TABLETS?|CAPSULES?|SYRUP"
    r")\b",
    re.IGNORECASE,
)

# 6. Perishable Foodstuff (Fresh fruit, mangoes, seafood, meat, cut flowers)
RE_PERISHABLES = re.compile(
    r"\b(?:"
    r"PERISHABLES?|PER|FOOD|FOODSTUFFS?|FRUITS?|VEGETABLES?|VEG|"
    r"MEAT|FRESH\s+FISH|SEAFOOD|CRABS?|LOBSTERS?|PRAWNS?|SHRIMPS?|"
    r"MANGOES?|FLOWERS?|CUT\s+FLOWERS?|FLORAL|DAIRY|CHEESE|PRODUCE|"
    r"CHILLED|FROZEN\s+FOOD|FROZEN\s+MEAT|ASPARAGUS|BERRIES|BERRY"
    r")\b",
    re.IGNORECASE,
)

# 7. Auto Parts (Brake pads, transmission, tires, auto accessories)
# Strict word boundary: \bAUTO\b avoids automation / automatic.
RE_AUTO_PARTS = re.compile(
    r"\b(?:"
    r"AUTO\s+PARTS?|AUTOMOTIVE(?:\s+PARTS?)?|CAR\s+PARTS?|MOTOR\s+PARTS?|VEHICLE\s+PARTS?|"
    r"AUTO\s+ACCESSOR(?:Y|IES)|AUTOMOBILE\s+SPARE\s+PARTS?|SPARE\s+PARTS?|"
    r"AUTO\s+COMPONENTS?|BRAKE\s+PADS?|CLUTCH|TRANSMISSION(?:\s+PARTS?)?|"
    r"SHOCK\s+ABSORBERS?|TIRES?|TYRES?|CAR\s+ENGINE|VEHICLE\s+ACCESSOR(?:Y|IES)|"
    r"AUTO"
    r")\b",
    re.IGNORECASE,
)

# 8. Garments & Textiles (Fabrics, cotton, carpets, apparel, leather footwear)
RE_GARMENTS = re.compile(
    r"\b(?:"
    r"GARMENTS?|TEXTILES?|APPAREL|CLOTHING|FABRICS?|COTTON|YARN|SILK|LEATHER|"
    r"CARPETS?|RUGS?|WOVEN|KNITTED|FOOTWEAR|SHOES?|DENIM|FIBER|FIBRE|SHIRTS?|TROUSERS?|LINEN"
    r")\b",
    re.IGNORECASE,
)

# 9. Engineering & Machinery (Turbines, pumps, valves, machinery, automation, circuit boards)
RE_ENGINEERING = re.compile(
    r"\b(?:"
    r"ENGINEERING|MACHINERY|MACHINES?|EQUIPMENT|HARDWARE|"
    r"TURBINES?|PUMPS?|GENERATORS?|CIRCUIT\s+BOARDS?|VALVES?|TRANSFORMERS?|"
    r"COMPRESSORS?|INDUSTRIAL|ELECTRONICS?|MECHANICAL|TOOLS?|TOOLING|"
    r"MOTORS?|AUTOMATION|ENGINES?|RADAR|SENSORS?|BOILERS?"
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
}


# ------------------------------------------------------------------------------
# REGIONAL WORD-BOUNDARY PATTERNS (KEYWORD FALLBACK)
# ------------------------------------------------------------------------------

REGION_PATTERNS = [
    ("Indian Subcontinent", re.compile(
        r"\b(?:MALE|COLOMBO|DHAKA|KARACHI|LAHORE|KATHMANDU|ISLAMABAD|CHITTAGONG|MLE|CMB|DAC|KHI|LHE|KTM|ISB|BANDARANAIKE|HAZRAT\s+SHAHJALAL)\b",
        re.IGNORECASE,
    )),
    ("Middle East", re.compile(
        r"\b(?:DUBAI|ZAYED|ABU\s+DHABI|SHARJAH|MUSCAT|DOHA|DAMMAM|JEDDAH|RIYADH|BAHRAIN|KUWAIT|SALALAH|AMMAN|BEIRUT|BAGHDAD|ERBIL|RAS\s+AL\s+KHAIMAH|TEL\s+AVIV|MEDINA|MADINAH|TEHRAN|DXB|AUH|DOH|JED|RUH)\b",
        re.IGNORECASE,
    )),
    ("Europe", re.compile(
        r"\b(?:HEATHROW|LONDON|FRANKFURT|AMSTERDAM|PARIS|CHARLES\s+DE\s+GAULLE|MILAN|MALPENSA|ROME|MADRID|BARCELONA|ZURICH|DUBLIN|BRUSSELS|BRUXELLES|VIENNA|MUNICH|MANCHESTER|BIRMINGHAM|ISTANBUL|ATHENS|WARSAW|PRAGUE|PRAHA|LISBON|LISBOA|BUDAPEST|STOCKHOLM|COPENHAGEN|OSLO|HAMBURG|WIEN|MOSCOW|STUTTGART|GENEVA|HELSINKI|DUSSELDORF|PORTO|FIUMICINO|BELGRADE|BRATISLAVA|NICE|GOTHENBURG|VILNIUS|MARSEILLE|SOFIA|MALAGA|ZAGREB|BERLIN|LYON|ROTTERDAM|LUXEMBOURG|LHR|FRA|CDG|AMS|MAD|BCN|MUC|FCO|IST)\b",
        re.IGNORECASE,
    )),
    ("North America", re.compile(
        r"\b(?:KENNEDY|NEW\s+YORK|JFK|PEARSON|TORONTO|MONTREAL|VANCOUVER|CHICAGO|LOS\s+ANGELES|LAX|ATLANTA|DALLAS|HOUSTON|MIAMI|SAN\s+FRANCISCO|BOSTON|SEATTLE|DETROIT|NEWARK|PHILADELPHIA|CALGARY|OTTAWA|DORVAL|CHARLOTTE|DULLES|WASHINGTON|DENVER|ORLANDO|COLUMBUS|PHOENIX|MEMPHIS|ORD|DFW|SFO|MIA|EWR|BOS|COLUMBIA)\b",
        re.IGNORECASE,
    )),
    ("Latin America", re.compile(
        r"\b(?:SAO\s+PAULO|BUENOS\s+AIRES|MEXICO(?:\s+CITY)?|BOGOTA|SANTIAGO|LIMA|PANAMA(?:\s+CITY)?|RIO(?:\s+DE\s+JANEIRO)?|QUITO|SANTO\s+DOMINGO|CARACAS|GUAYAQUIL|ASUNCION|MONTERREY|VIRACOPOS|CURITIBA|SAN\s+SALVADOR|MONTEVIDEO|GUADALAJARA|CALI|MEDELLIN|CANCUN|CARTAGEN(?:A)?|GRU|GIG|BOG|EZE|SCL|LIM|PTY|EL\s+EDEN)\b",
        re.IGNORECASE,
    )),
    ("Africa", re.compile(
        r"\b(?:CAIRO|LAGOS|NAIROBI|KENYATTA|JOHANNESBURG|OR\s+TAMBO|ADDIS\s+ABABA|ACCRA|ENTEBBE|DAR\s+ES\s+SALAAM|KIGALI|CASABLANCA|HARARE|LUSAKA|ALGIERS|TUNIS|ABIDJAN|DAKAR|MAPUTO|LUANDA|MAURITIUS|DURBAN|CAPE\s+TOWN|YAOUNDE|JNB|CAI|NBO|LOS|ADD|CPT|BERENICE)\b",
        re.IGNORECASE,
    )),
    ("Asia-Pacific", re.compile(
        r"\b(?:SINGAPORE|SHANGHAI|PUDONG|BANGKOK|SUVARNABHUMI|HONG\s+KONG|TOKYO|NARITA|HANEDA|SEOUL|INCHEON|KUALA\s+LUMPUR|JAKARTA|MANILA|SYDNEY|MELBOURNE|HANOI|HO\s+CHI\s+MINH|TAIPEI|GUANGZHOU|BEIJING|AUCKLAND|PERTH|BRISBANE|SIN|PVG|BKK|HKG|NRT|HND|ICN|KUL|CGK|MNL|SYD|MEL|YAO)\b",
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
