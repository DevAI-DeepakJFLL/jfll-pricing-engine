"""
tests/test_taxonomy.py
Unit tests for Word-Boundary Commodity Classification and
Country-First Destination Region Mapping (Fixes Issues C1, C2, D7).
"""

import unittest
from src.taxonomy import (
    classify_commodity,
    map_destination_region,
    COMMODITY_CATEGORIES,
    DESTINATION_REGIONS,
)


class TestCommodityWordBoundaries(unittest.TestCase):
    """
    Verifies that regex word boundaries (\\b) strictly prevent false substring
    collisions (Issue C1: PET vs petroleum/carpet, AVI vs aviation, AUTO vs automation, etc.).
    """

    def test_pet_word_boundary_prevents_collisions(self):
        # Genuine Live Animals matching PET
        self.assertEqual(classify_commodity("LIVE PET DOG"), "Live Animals")
        self.assertEqual(classify_commodity("PET CAT TRANSPORT"), "Live Animals")
        self.assertEqual(classify_commodity("DOMESTIC PETS"), "Live Animals")

        # False hits previously triggered by substring 'PET'
        self.assertNotEqual(classify_commodity("PETROL ENGINE PART"), "Live Animals")
        self.assertNotEqual(classify_commodity("PETROLEUM DRUMS"), "Live Animals")
        self.assertNotEqual(classify_commodity("WOVEN CARPET RUNNERS"), "Live Animals")
        self.assertEqual(classify_commodity("WOVEN CARPET RUNNERS"), "Garments / Textiles")

    def test_avi_word_boundary_prevents_collisions(self):
        # Genuine Live Animals matching IATA special cargo code AVI
        self.assertEqual(classify_commodity("AVI CARGO - 2 DOGS"), "Live Animals")
        self.assertEqual(classify_commodity("AVI SHIPMENT"), "Live Animals")

        # False hits previously triggered by substring 'AVI'
        self.assertNotEqual(classify_commodity("AVIATION SPARE PARTS"), "Live Animals")
        self.assertNotEqual(classify_commodity("AVIONICS RADAR SYSTEM"), "Live Animals")
        self.assertNotEqual(classify_commodity("NAVIGATING EQUIPMENT"), "Live Animals")
        self.assertNotEqual(classify_commodity("HEAVY CAVIAR SHIPMENT"), "Live Animals")

    def test_un_number_vs_generic_chemical_and_plain_un(self):
        # Genuine Dangerous Goods matching UN 4-digit code
        self.assertEqual(classify_commodity("UN 1234 CHEMICAL SAMPLES"), "Dangerous Goods")
        self.assertEqual(classify_commodity("UN3480 LITHIUM BATTERIES"), "Dangerous Goods")
        self.assertEqual(classify_commodity("HAZARDOUS UN 1993 FLAMMABLE"), "Dangerous Goods")

        # Generic chemical or non-DG must NOT match Dangerous Goods
        self.assertNotEqual(classify_commodity("ORGANIC CHEMICAL POWDER"), "Dangerous Goods")
        self.assertNotEqual(classify_commodity("CHEMICAL SAMPLES NON DG"), "Dangerous Goods")
        self.assertNotEqual(classify_commodity("NON-DG INDUSTRIAL CHEMICAL"), "Dangerous Goods")
        self.assertNotEqual(classify_commodity("NOT RESTRICTED CHEMICAL COMPOUND"), "Dangerous Goods")

        # Plain UN without 4 digits must NOT match Dangerous Goods
        self.assertNotEqual(classify_commodity("UN DIPLOMATIC POUCH"), "Dangerous Goods")
        self.assertNotEqual(classify_commodity("UNITED NATIONS RELIEF SUPPLIES"), "Dangerous Goods")

    def test_auto_word_boundary_prevents_collisions(self):
        # Genuine Auto Parts
        self.assertEqual(classify_commodity("AUTO SPARE PARTS"), "Auto Parts")
        self.assertEqual(classify_commodity("AUTO ACCESSORIES"), "Auto Parts")
        self.assertEqual(classify_commodity("AUTOMOTIVE BRAKE PADS"), "Auto Parts")

        # False hits previously triggered by substring 'AUTO'
        self.assertNotEqual(classify_commodity("AUTOMATION ROBOTIC CONTROLLER"), "Auto Parts")
        self.assertEqual(classify_commodity("AUTOMATION ROBOTIC CONTROLLER"), "Engineering & Machinery")

        self.assertNotEqual(classify_commodity("AUTOMATIC PACKAGING MACHINE"), "Auto Parts")
        self.assertEqual(classify_commodity("AUTOMATIC PACKAGING MACHINE"), "Engineering & Machinery")

        self.assertNotEqual(classify_commodity("LUXURY AUTOMATIC WATCH"), "Auto Parts")
        self.assertEqual(classify_commodity("LUXURY AUTOMATIC WATCH"), "Valuables")

    def test_dog_cat_horse_word_boundaries(self):
        # Word boundaries prevent false collisions in English vocabulary
        self.assertEqual(classify_commodity("LIVE DOG TRANSPORT"), "Live Animals")
        self.assertEqual(classify_commodity("DOMESTIC CAT"), "Live Animals")
        self.assertEqual(classify_commodity("RACE HORSE"), "Live Animals")

        self.assertNotEqual(classify_commodity("DOGMA PHILOSOPHY BOOKS"), "Live Animals")
        self.assertNotEqual(classify_commodity("PRINTED CATALOG SAMPLES"), "Live Animals")
        self.assertNotEqual(classify_commodity("UNHORSED SADDLE LEATHER"), "Live Animals")

    def test_per_preposition_does_not_trigger_perishables(self):
        # Commercial prepositions 'as per', 'per piece', 'per contract' must not trigger Perishables
        self.assertEqual(classify_commodity("MACHINERY SPARE PARTS AS PER PACKING LIST"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("RATE AS PER CONTRACT - GENERAL GOODS"), "General Cargo")
        self.assertEqual(classify_commodity("COTTON SHIRTS AS PER INVOICE"), "Garments / Textiles")
        self.assertEqual(classify_commodity("10 USD PER PIECE - ELECTRONIC SENSORS"), "Engineering & Machinery")

    def test_spare_parts_scoping_auto_vs_engineering(self):
        # Automotive spare parts -> Auto Parts
        self.assertEqual(classify_commodity("AUTO SPARE PARTS"), "Auto Parts")
        self.assertEqual(classify_commodity("AUTOMOBILE SPARE PARTS"), "Auto Parts")
        self.assertEqual(classify_commodity("CAR SPARE PARTS"), "Auto Parts")

        # Aviation, industrial, and machinery spare parts -> Engineering & Machinery
        self.assertEqual(classify_commodity("AVIATION SPARE PARTS"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("AIRCRAFT SPARE PARTS"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("PUMP SPARE PARTS"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("MACHINE SPARE PARTS"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("INDUSTRIAL SPARE PARTS"), "Engineering & Machinery")

    def test_fiber_and_tablet_scoping(self):
        # Optical and carbon fiber -> Engineering & Machinery (NOT Garments)
        self.assertEqual(classify_commodity("OPTICAL FIBER CABLE"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("CARBON FIBER PANELS"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("COTTON FIBER YARN"), "Garments / Textiles")

        # Computing tablets -> Engineering & Machinery (NOT Pharmaceuticals)
        self.assertEqual(classify_commodity("TABLET COMPUTER"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("GRAPHICS TABLET"), "Engineering & Machinery")

        # Medicinal tablets -> Pharmaceuticals
        self.assertEqual(classify_commodity("PHARMACEUTICAL TABLETS"), "Pharmaceuticals")
        self.assertEqual(classify_commodity("PARACETAMOL TABLETS"), "Pharmaceuticals")


class TestRestoredCommodityClasses(unittest.TestCase):
    """
    Verifies that Special Cargo classes (Dangerous Goods, Valuables, Live Animals)
    and standard categories are properly recognized and restored (Issue D7).
    """

    def test_dangerous_goods_restoration(self):
        self.assertEqual(classify_commodity("HAZMAT CLASS 9 MISCELLANEOUS"), "Dangerous Goods")
        self.assertEqual(classify_commodity("LITHIUM ION BATTERIES"), "Dangerous Goods")
        self.assertEqual(classify_commodity("FLAMMABLE LIQUID CLASS 3"), "Dangerous Goods")
        self.assertEqual(classify_commodity("RADIOACTIVE MEDICAL ISOTOPES"), "Dangerous Goods")

    def test_valuables_restoration(self):
        self.assertEqual(classify_commodity("VALUABLE GOLD JEWELLERY"), "Valuables")
        self.assertEqual(classify_commodity("SILVER BULLION 999"), "Valuables")
        self.assertEqual(classify_commodity("CUT DIAMONDS AND PRECIOUS GEMS"), "Valuables")
        self.assertEqual(classify_commodity("FOREIGN CURRENCY BANKNOTES"), "Valuables")
        self.assertEqual(classify_commodity("HIGH END WRIST WATCHES"), "Valuables")

    def test_live_animals_restoration(self):
        self.assertEqual(classify_commodity("DAY OLD CHICKS IN CRATES"), "Live Animals")
        self.assertEqual(classify_commodity("ORNAMENTAL TROPICAL FISH"), "Live Animals")
        self.assertEqual(classify_commodity("LIVE EQUINE HORSES"), "Live Animals")

    def test_pharmaceuticals(self):
        self.assertEqual(classify_commodity("COVID-19 VACCINES TEMPERATURE CONTROLLED"), "Pharmaceuticals")
        self.assertEqual(classify_commodity("PHARMACEUTICAL MEDICINES"), "Pharmaceuticals")
        self.assertEqual(classify_commodity("DIAGNOSTIC KITS AND REAGENTS"), "Pharmaceuticals")
        self.assertEqual(classify_commodity("INSULIN VIALS"), "Pharmaceuticals")

    def test_perishable_foodstuff(self):
        self.assertEqual(classify_commodity("FRESH ALPHONSO MANGOES"), "Perishable Foodstuff")
        self.assertEqual(classify_commodity("CHILLED FRESH FISH AND SEAFOOD"), "Perishable Foodstuff")
        self.assertEqual(classify_commodity("FRESH CUT FLOWERS"), "Perishable Foodstuff")
        self.assertEqual(classify_commodity("FROZEN MEAT CARCASSES"), "Perishable Foodstuff")

    def test_engineering_and_machinery(self):
        self.assertEqual(classify_commodity("HYDRAULIC TURBINE GENERATOR"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("INDUSTRIAL VALVES AND PUMPS"), "Engineering & Machinery")
        self.assertEqual(classify_commodity("ELECTRONIC CIRCUIT BOARDS"), "Engineering & Machinery")

    def test_garments_and_textiles(self):
        self.assertEqual(classify_commodity("COTTON KNITTED APPAREL"), "Garments / Textiles")
        self.assertEqual(classify_commodity("SILK FABRIC ROLLS"), "Garments / Textiles")
        self.assertEqual(classify_commodity("LEATHER FOOTWEAR AND JACKETS"), "Garments / Textiles")

    def test_courier(self):
        self.assertEqual(classify_commodity("EXPRESS COURIER DOCUMENTS"), "Courier")
        self.assertEqual(classify_commodity("COMMERCIAL SAMPLES POUCH"), "Courier")

    def test_general_cargo_and_fallback(self):
        self.assertEqual(classify_commodity("GENERAL CARGO CONSOLIDATION"), "General Cargo")
        self.assertEqual(classify_commodity("FREIGHT ALL KINDS (FAK)"), "General Cargo")
        self.assertEqual(classify_commodity("HOUSEHOLD GOODS AND PERSONAL EFFECTS"), "General Cargo")
        self.assertEqual(classify_commodity(""), "General Cargo")
        self.assertEqual(classify_commodity(None), "General Cargo")
        self.assertEqual(classify_commodity("MISCELLANEOUS SPECIAL GOODS"), "Other")


class TestDestinationRegionMapping(unittest.TestCase):
    """
    Verifies country-first regional resolution and disambiguation of tricky ports
    ('Nice', 'Male', 'Armenia', 'Cali', 'Yaounde') to prevent substring collisions (Issue C2).
    """

    def test_armenia_country_vs_colombia_disambiguation(self):
        # Armenia as sovereign nation (EVN / Yerevan gateway) -> Europe
        self.assertEqual(map_destination_region("Yerevan", country_name="Armenia"), "Europe")
        self.assertEqual(map_destination_region("Armenia", country_name="Armenia"), "Europe")
        self.assertEqual(map_destination_region("Armenia"), "Europe")

        # Armenia as city/port in Quindio, Colombia (AXM / El Eden) -> Latin America
        self.assertEqual(map_destination_region("Armenia", country_name="Colombia"), "Latin America")
        self.assertEqual(map_destination_region("El Eden Airport", country_name="Colombia"), "Latin America")
        self.assertEqual(map_destination_region("Armenia, Colombia"), "Latin America")

    def test_nice_france_without_false_matching(self):
        # Nice Côte d'Azur (NCE, France) -> Europe
        self.assertEqual(map_destination_region("Nice", country_name="France"), "Europe")
        self.assertEqual(map_destination_region("Nice"), "Europe")
        self.assertEqual(map_destination_region("Nice Côte d'Azur Airport"), "Europe")
        self.assertEqual(map_destination_region("NCE"), "Europe")

        # String containing 'nice' in another country must NOT resolve to Europe
        self.assertEqual(map_destination_region("Berenice", country_name="Egypt"), "Africa")
        self.assertEqual(map_destination_region("Port of Berenice", country_name="Egypt"), "Africa")

    def test_male_maldives_without_false_matching(self):
        # Male / Velana International Airport (MLE, Maldives) -> Indian Subcontinent
        self.assertEqual(map_destination_region("Male", country_name="Maldives"), "Indian Subcontinent")
        self.assertEqual(map_destination_region("Male"), "Indian Subcontinent")
        self.assertEqual(map_destination_region("Velana International Airport", country_name="Maldives"), "Indian Subcontinent")
        self.assertEqual(map_destination_region("MLE"), "Indian Subcontinent")

        # Ports containing 'male' substring in Latin America or Africa must NOT collide with Male
        self.assertEqual(map_destination_region("Guatemala City", country_name="Guatemala"), "Latin America")
        self.assertEqual(map_destination_region("La Aurora Airport", country_name="Guatemala"), "Latin America")
        self.assertEqual(map_destination_region("Tamale", country_name="Ghana"), "Africa")

    def test_other_tricky_ports(self):
        # Cali (Colombia, Latin America) vs California (North America)
        self.assertEqual(map_destination_region("Cali", country_name="Colombia"), "Latin America")
        self.assertEqual(map_destination_region("Columbia", country_name="United States"), "North America")

        # Yaounde (Cameroon, Africa) vs Yao (Japan, Asia-Pacific)
        self.assertEqual(map_destination_region("Yaounde", country_name="Cameroon"), "Africa")
        self.assertEqual(map_destination_region("Yao", country_name="Japan"), "Asia-Pacific")

    def test_standard_seven_regions_coverage(self):
        # Verify major global commercial air export gateways resolve to 7 standard regions
        gateways = [
            ("Heathrow Apt/London", "United Kingdom", "Europe"),
            ("Frankfurt", "Germany", "Europe"),
            ("Dubai", "United Arab Emirates", "Middle East"),
            ("Doha", "Qatar", "Middle East"),
            ("John F Kennedy Int Apt/New York", "United States", "North America"),
            ("Chicago O'Hare", "United States", "North America"),
            ("Singapore Changi", "Singapore", "Asia-Pacific"),
            ("Tokyo Narita", "Japan", "Asia-Pacific"),
            ("Sao Paulo Guarulhos", "Brazil", "Latin America"),
            ("Bogota El Dorado", "Colombia", "Latin America"),
            ("O.R. Tambo / Johannesburg", "South Africa", "Africa"),
            ("Nairobi Kenyatta", "Kenya", "Africa"),
            ("Bandaranaike / Colombo", "Sri Lanka", "Indian Subcontinent"),
            ("Hazrat Shahjalal / Dhaka", "Bangladesh", "Indian Subcontinent"),
        ]
        for port, country, expected_region in gateways:
            resolved = map_destination_region(port, country_name=country)
            self.assertEqual(
                resolved,
                expected_region,
                f"Failed for port='{port}', country='{country}': expected {expected_region}, got {resolved}"
            )
            self.assertIn(resolved, DESTINATION_REGIONS)

    def test_unmapped_port_fallback(self):
        self.assertEqual(map_destination_region("Unknown Nonexistent Gateway"), "Other Destination")


class TestTaxonomyConstants(unittest.TestCase):
    """
    Verifies that exported taxonomy constants conform to the required vocabulary.
    """

    def test_commodity_categories_constant(self):
        expected_categories = [
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
        self.assertEqual(COMMODITY_CATEGORIES, expected_categories)
        self.assertEqual(len(COMMODITY_CATEGORIES), 11)

    def test_destination_regions_constant(self):
        expected_regions = [
            "Europe",
            "Middle East",
            "North America",
            "Asia-Pacific",
            "Latin America",
            "Africa",
            "Indian Subcontinent",
        ]
        self.assertEqual(DESTINATION_REGIONS, expected_regions)
        self.assertEqual(len(DESTINATION_REGIONS), 7)


if __name__ == "__main__":
    unittest.main()
