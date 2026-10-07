"""
tests/test_preprocess_validation.py
Unit tests for physical plausibility validation, volumetric calculations, and outlier quarantine.
"""

import unittest
from src.preprocess import (
    validate_shipment_physics,
    compute_volumetric_weight,
    clean_incoterms,
    map_commodity_group,
)


class TestPreprocessValidation(unittest.TestCase):
    def test_compute_volumetric_weight_single_box(self):
        # 100cm x 50cm x 40cm, 1 piece -> 200,000 / 6000 = 33.333 kg
        dims = [{"length": 100, "width": 50, "height": 40, "pieces": 1}]
        vol = compute_volumetric_weight(dims)
        self.assertAlmostEqual(vol, 33.33, places=1)

    def test_compute_volumetric_weight_multi_piece(self):
        # 2 boxes of 60x50x40 -> 2 * 120,000 / 6000 = 40.0 kg
        dims = [{"length": 60, "width": 50, "height": 40, "pieces": 2}]
        vol = compute_volumetric_weight(dims)
        self.assertAlmostEqual(vol, 40.0, places=2)

    def test_validate_physics_normal_dense_cargo(self):
        # Gross 120kg, Vol 80kg -> Ch Wt = 120kg, Density = 1.0 (or bounded <= 1.0)
        valid, ch_wt, density, err = validate_shipment_physics(
            gross_wt=120.0,
            ch_wt=120.0,
            dimensions=[{"length": 60, "width": 50, "height": 40, "pieces": 4}] # 80kg vol
        )
        self.assertTrue(valid)
        self.assertEqual(ch_wt, 120.0)
        self.assertAlmostEqual(density, 1.0, places=2)
        self.assertIsNone(err)

    def test_validate_physics_volumetric_cargo(self):
        # Gross 50kg, Vol 80kg -> Ch Wt = 80kg, Density = 50/80 = 0.625
        valid, ch_wt, density, err = validate_shipment_physics(
            gross_wt=50.0,
            ch_wt=50.0,
            dimensions=[{"length": 60, "width": 50, "height": 40, "pieces": 4}] # 80kg vol
        )
        self.assertTrue(valid)
        self.assertEqual(ch_wt, 80.0)
        self.assertAlmostEqual(density, 0.625, places=2)
        self.assertIsNone(err)

    def test_validate_physics_gross_greater_than_chargeable_repaired(self):
        # Gross 100kg, but reported Ch Wt 80kg (impossible under IATA). Should repair to 100kg.
        valid, ch_wt, density, err = validate_shipment_physics(
            gross_wt=100.0,
            ch_wt=80.0
        )
        self.assertTrue(valid)
        self.assertEqual(ch_wt, 100.0)
        self.assertEqual(density, 1.0)

    def test_validate_physics_outlier_quarantine(self):
        # Extreme weight <= 0.5kg
        valid, _, _, err = validate_shipment_physics(gross_wt=0.2, ch_wt=0.2)
        self.assertFalse(valid)
        self.assertIn("minimum weight", err.lower())

        # Extreme weight > 25,000kg
        valid, _, _, err = validate_shipment_physics(gross_wt=35000.0, ch_wt=35000.0)
        self.assertFalse(valid)
        self.assertIn("maximum commercial weight", err.lower())

    def test_clean_incoterms(self):
        self.assertEqual(clean_incoterms("FOB"), "FOB")
        self.assertEqual(clean_incoterms("CIF Mumbai"), "CIF")
        self.assertEqual(clean_incoterms("EXW Delhi"), "EXW")
        self.assertEqual(clean_incoterms("DAP Berlin"), "DAP/DDP")
        self.assertEqual(clean_incoterms(None), "FOB")

    def test_map_commodity_group_expanded(self):
        self.assertEqual(map_commodity_group("PHARMACEUTICAL VACCINES IN COLD BOX"), "Pharmaceuticals")
        self.assertEqual(map_commodity_group("DANGEROUS GOODS CLASS 9 LITHIUM"), "Dangerous Goods")
        self.assertEqual(map_commodity_group("PERISHABLE FRESH MANGOES"), "Perishable Foodstuff")
        self.assertEqual(map_commodity_group("AUTOMOTIVE ENGINE OIL PUMP"), "Auto Parts")
        self.assertEqual(map_commodity_group("HEAVY INDUSTRIAL CNC MACHINERY"), "Engineering & Machinery")
        self.assertEqual(map_commodity_group("COTTON T-SHIRTS AND APPAREL"), "Garments / Textiles")
        self.assertEqual(map_commodity_group("COURIER SAMPLES"), "Courier")
        self.assertEqual(map_commodity_group("GENERAL CARGO NON DG"), "General Cargo")


if __name__ == "__main__":
    unittest.main()
