import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arr_aggregate import factorial_aggregate, route_aggregate
from arr_local_update_check import interpolate_score_from_pre_snapshot
import torch


class AggregationChecks(unittest.TestCase):
    def test_local_update_interpolation_restores_post_step_and_reports_quantization(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        pre = {"weight": torch.tensor([[1.0]])}
        rows, meta = interpolate_score_from_pre_snapshot(
            model, pre, lambda m: float(m.weight.detach().pow(2).sum()), epsilon=1e-3)
        self.assertEqual([r["scale"] for r in rows], [0.0, .25, .5, 1.0])
        self.assertAlmostEqual(rows[0]["score_at_zero"], 1.0, places=6)
        self.assertAlmostEqual(rows[0]["directional_derivative"], 2.0, places=3)
        self.assertAlmostEqual(rows[-1]["observed_change"], 3.0, places=6)
        self.assertEqual(rows[-1]["changed_parameter_count"], 1)
        self.assertAlmostEqual(rows[-1]["max_coordinate_perturbation"], 1.0, places=6)
        self.assertEqual(meta["score_evaluation_scale_sequence"], [0.0, .25, .5, 1.0, -1e-3, 1e-3])
        self.assertTrue(torch.equal(model.weight.detach(), torch.tensor([[2.0]])))

    def test_factorial_shared_pairs_and_interaction(self):
        rows = []
        means = {"E": 0.0, "P": -1.0, "W": -2.0, "P+W": -4.0,
                 "wrong_region_mix": -1.0, "global_mix": -3.0, "slowdown": -1.0}
        for c, v in means.items():
            for i in range(12):
                rows.append({"condition": c, "prompt_id": str(i), "cluster_id": str(i),
                             "value": v, "outcome": "MR"})
        out = factorial_aggregate(rows, draws=50, seed=1, expected_ids=[str(i) for i in range(12)])
        got = {r["contrast"]: r["estimate"] for r in out}
        self.assertEqual(got["interaction"], -1.0)
        self.assertEqual(got["P_at_W0"], -1.0)
        self.assertEqual(got["W_at_P0"], -2.0)
        rows = [r for r in rows if r["condition"] in ("E", "W")]
        partial = factorial_aggregate(rows, draws=20, seed=1, expected_ids=[str(i) for i in range(12)])
        missing = next(r for r in partial if r["contrast"] == "P_at_W0")
        self.assertEqual(missing["estimate"], "")
        self.assertIn("condition data missing: P", missing["null_reason"])

    def test_factorial_requires_frozen_id_set_and_consistent_clusters(self):
        rows = [{"condition": c, "prompt_id": "p", "cluster_id": cid,
                 "value": 0, "outcome": "MR"} for c, cid in (("E", "a"), ("W", "b"))]
        with self.assertRaisesRegex(ValueError, "cluster ID changes"):
            factorial_aggregate(rows, draws=10, expected_ids=["p"])
        rows[1]["cluster_id"] = "a"
        with self.assertRaisesRegex(ValueError, "expected prompt IDs"):
            factorial_aggregate(rows, draws=10, expected_ids=["p", "missing"])

    def test_route_identity_correction_and_observed_ratio(self):
        rows = []
        # 3 prompts have T values 1, 2, 9 and M values .2, .4, 1.8.
        # The point ratio is 0.2; a ratio of bootstrap mean estimates is not
        # used in place of the observed numerator/denominator contrast.
        for i, (t, m, baseline_shift) in enumerate(((1, .2, .1), (2, .4, 0), (9, 1.8, -.1))):
            c = 0.0; g = c+t; hc = c+baseline_shift; hg = hc+(t-m)
            rows.append({"run_id":"r", "model":"m", "seed":1, "rendering":"x",
                         "step":2, "condition":"E", "subset":"N", "prompt_id":str(i),
                         "cluster_id":str(i), "score_c":c, "score_x":10,
                         "score_g":g, "score_hg":hg, "score_hc":hc})
        out = route_aggregate(rows, draws=100, seed=4, expected_ids=["0", "1", "2"])
        got = {r["outcome"]: r for r in out}
        self.assertAlmostEqual(got["T"]["estimate"], 4.0)
        self.assertAlmostEqual(got["D"]["estimate"], 3.2)
        self.assertAlmostEqual(got["M"]["estimate"], .8)
        self.assertAlmostEqual(got["removed_fraction"]["estimate"], .2)
        self.assertAlmostEqual(got["identity_baseline_error"]["estimate"], 0.0)
        self.assertAlmostEqual(got["decomposition_arithmetic_residual"]["estimate"], 0.0)
        self.assertAlmostEqual(got["S_C"]["estimate"], 0.0)
        self.assertAlmostEqual(got["S_G"]["estimate"], 4.0)
        with self.assertRaisesRegex(ValueError, "expected route prompt IDs"):
            route_aggregate(rows[:-1], draws=10, expected_ids=["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()
