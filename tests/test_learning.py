from __future__ import annotations

import base64
import copy
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.learning import (
    CausalGate,
    DriftDetector,
    FeatureBaseline,
    LearnedDecision,
    ModelTrustError,
    detect_change_point,
    verify_model_manifest,
)
from kval_node.signing import canonical_json


def manifest_body() -> dict:
    return {
        "model_id": "link-failure-forecaster-v1",
        "engine_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "calibration_report_sha256": "c" * 64,
        "training_data_lineage": "kval-field-traces-2026-07",
        "created_at": "2026-07-15T00:00:00+00:00",
    }


def sign_manifest(body: dict, key: Ed25519PrivateKey) -> dict:
    from kval_node.learning import ModelManifest
    from kval_node.signing import public_key_id

    manifest = ModelManifest.from_dict(body)
    signature = key.sign(canonical_json(manifest.canonical()))
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(key.public_key()),
        "manifest": manifest.canonical(),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


class ModelManifestTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        key = Ed25519PrivateKey.generate()
        envelope = sign_manifest(manifest_body(), key)
        manifest = verify_model_manifest(envelope, key.public_key())
        self.assertEqual(manifest.model_id, "link-failure-forecaster-v1")

    def test_tamper_fails_closed(self) -> None:
        key = Ed25519PrivateKey.generate()
        envelope = sign_manifest(manifest_body(), key)
        altered = copy.deepcopy(envelope)
        altered["manifest"]["engine_sha256"] = "d" * 64
        with self.assertRaises(ModelTrustError):
            verify_model_manifest(altered, key.public_key())

    def test_incomplete_manifest_fails_closed(self) -> None:
        key = Ed25519PrivateKey.generate()
        body = manifest_body()
        del body["calibration_report_sha256"]
        with self.assertRaises(ModelTrustError):
            verify_model_manifest(sign_manifest_missing(body, key), key.public_key())


def sign_manifest_missing(body: dict, key: Ed25519PrivateKey) -> dict:
    from kval_node.signing import public_key_id

    signature = key.sign(canonical_json(body))
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(key.public_key()),
        "manifest": body,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


class DriftDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = FeatureBaseline.from_dict(
            {"means": {"rtt": 50.0, "loss": 1.0}, "stddevs": {"rtt": 10.0, "loss": 0.5}}
        )

    def test_in_distribution_advises(self) -> None:
        verdict = DriftDetector(self.baseline).evaluate({"rtt": 55.0, "loss": 1.2})
        self.assertIs(verdict.decision, LearnedDecision.ADVISE)

    def test_out_of_distribution_abstains(self) -> None:
        verdict = DriftDetector(self.baseline).evaluate({"rtt": 500.0, "loss": 1.0})
        self.assertIs(verdict.decision, LearnedDecision.ABSTAIN)
        self.assertIn("rtt", verdict.out_of_range_features)

    def test_missing_feature_abstains(self) -> None:
        verdict = DriftDetector(self.baseline).evaluate({"rtt": 50.0})
        self.assertIs(verdict.decision, LearnedDecision.ABSTAIN)
        self.assertIn("loss", verdict.missing_features)


class ChangePointTests(unittest.TestCase):
    def test_detects_shift(self) -> None:
        series = [1.0] * 10 + [50.0] * 10
        result = detect_change_point(series, threshold_sigmas=3.0)
        self.assertTrue(result.detected)

    def test_stable_series_reports_no_change(self) -> None:
        result = detect_change_point([5.0, 5.1, 4.9, 5.0, 5.05, 4.95, 5.0, 5.02])
        self.assertFalse(result.detected)

    def test_short_series_insufficient(self) -> None:
        result = detect_change_point([1.0, 2.0])
        self.assertFalse(result.detected)
        self.assertIn("INSUFFICIENT_SAMPLES", result.reasons)


class CausalGateTests(unittest.TestCase):
    def test_clear_improvement_attributed(self) -> None:
        gate = CausalGate(min_effect=5.0, t_threshold=2.0)
        before = [100.0, 102.0, 98.0, 101.0, 99.0]
        after = [10.0, 12.0, 9.0, 11.0, 10.5]
        verdict = gate.evaluate(before, after, higher_is_better=False)
        self.assertTrue(verdict.attributed)

    def test_noise_not_attributed(self) -> None:
        gate = CausalGate(min_effect=5.0, t_threshold=2.0)
        before = [100.0, 102.0, 98.0, 101.0, 99.0]
        after = [100.5, 101.5, 98.5, 100.0, 99.5]
        verdict = gate.evaluate(before, after, higher_is_better=False)
        self.assertFalse(verdict.attributed)

    def test_insufficient_samples(self) -> None:
        gate = CausalGate(min_effect=1.0)
        verdict = gate.evaluate([1.0, 2.0], [3.0, 4.0], higher_is_better=True)
        self.assertFalse(verdict.attributed)
        self.assertIn("INSUFFICIENT_SAMPLES", verdict.reasons)


if __name__ == "__main__":
    unittest.main()
