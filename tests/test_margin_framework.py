import os, sys, math, unittest
import numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import margin_framework as mf
except ImportError:
    from src import margin_framework as mf

CSV = os.environ.get("PRICING_CSV", "data/processed/Air_Export_Pricing_Combined_ML.csv")


class TestMarginFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(CSV)
        cls.won = df[df.is_won == 1]
        cls.t = mf.build_tables(cls.won)

    def q(self, **kw):
        base = dict(Total_Buy_INR=125000, Chargeable_Weight_Kg=500, Destination_Region="Middle East",
                    Commodity_Group="General Cargo", Company_Group="Subagent", Air_Cargo_Season="Q2_Perishable_Peak")
        base.update(kw); return mf.recommend(base, self.t)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(mf.ADJUSTER_WEIGHTS.values()), 1.0, places=6)

    def test_tiers_strictly_ordered_and_floor_respected(self):
        for kw in [{}, dict(Commodity_Group="Pharmaceuticals"), dict(Total_Buy_INR=5000, Chargeable_Weight_Kg=20),
                   dict(Commodity_Group="Dangerous Goods", Company_Group="GSA")]:
            o = self.q(**kw); t = o["tiers"]
            self.assertLess(t["floor"]["margin_inr"], t["balanced"]["margin_inr"])
            self.assertLess(t["balanced"]["margin_inr"], t["premium"]["margin_inr"])
            self.assertGreaterEqual(t["floor"]["margin_inr"] + 0.01, o["cost_floor_inr"])

    def test_cap_never_exceeded_unless_floor_conflicts(self):
        o = self.q(Company_Group="GSA"); buy = 125000
        for v in o["tiers"].values():
            if v["margin_inr"] > o["cost_floor_inr"] + 1:
                self.assertLessEqual(v["margin_pct"], o["cap_pct"] + 0.01)

    def test_balanced_within_historical_band_for_general_cargo(self):
        o = self.q(); self.assertGreaterEqual(o["tiers"]["balanced"]["margin_per_kg"], 0.5 * o["band"]["P25"])
        self.assertLessEqual(o["tiers"]["balanced"]["margin_per_kg"], o["band"]["P85"])

    def test_sublinear_cost_pass_through(self):
        a = self.q()["tiers"]["balanced"]["margin_inr"]; b = self.q(Total_Buy_INR=150000)["tiers"]["balanced"]["margin_inr"]
        growth = b / a - 1
        self.assertGreater(growth, 0.0); self.assertLess(growth, 0.20)      # +20% buy -> clearly less than +20% margin

    def test_commodity_and_category_move_the_quote_in_expected_direction(self):
        gen = self.q()["tiers"]["balanced"]["margin_pct"]
        self.assertGreater(self.q(Commodity_Group="Dangerous Goods")["tiers"]["balanced"]["margin_pct"], gen)
        self.assertLess(self.q(Company_Group="GSA")["tiers"]["balanced"]["margin_pct"], gen)

    def test_special_cargo_requires_approval(self):
        self.assertTrue(self.q(Commodity_Group="Dangerous Goods")["approval_required"])
        self.assertTrue(self.q(Commodity_Group="Valuables")["approval_required"])

    def test_deterministic(self):
        self.assertEqual(self.q()["tiers"], self.q()["tiers"])

    def test_history_percentile_monotonic(self):
        slab = "500-1000kg"; xs = [1, 5, 10, 20, 50]
        ps = [mf.history_percentile(self.t, slab, x) for x in xs]
        self.assertEqual(ps, sorted(ps))

    def test_legacy_response_has_ui_keys_and_no_fake_win_prob(self):
        inq = dict(Total_Buy_INR=125000, Chargeable_Weight_Kg=500, Destination_Region="Middle East",
                   Commodity_Group="General Cargo", Company_Group="Subagent")
        rec = mf.recommend(inq, self.t); out = mf.to_legacy_response(inq, rec, self.t)
        for k in ("floor", "balanced", "premium"):
            s = out["strategic_recommendations"][k]
            for key in ("Margin_Percentage", "Quoted_Sell_Price_INR", "Margin_Amount_INR", "Strategy_Name", "Badge_Label"):
                self.assertIn(key, s)
            self.assertIsNone(s["Win_Probability"])
        self.assertEqual(len(mf.sensitivity_grid(inq, rec, self.t)), 15)

    def test_spot_tender_mode_competitiveness_and_floor_protection(self):
        std = self.q()
        tender = self.q(is_spot_tender=True)
        self.assertEqual(tender["pricing_mode"], "spot_tender")
        self.assertLessEqual(tender["tiers"]["balanced"]["margin_inr"], std["tiers"]["balanced"]["margin_inr"])
        self.assertGreaterEqual(tender["tiers"]["floor"]["margin_inr"] + 0.01, tender["cost_floor_inr"])
        self.assertGreaterEqual(tender["tiers"]["balanced"]["margin_inr"] + 0.01, tender["cost_floor_inr"])


if __name__ == "__main__":
    unittest.main()
