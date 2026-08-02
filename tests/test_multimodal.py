from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from kval_node.multimodal import Modality, ModalityObservation, MultimodalFuser

NOW = datetime.now(timezone.utc)


def make(modality: Modality, *, observed_at=NOW, features=None) -> ModalityObservation:
    return ModalityObservation.create(
        modality,
        source_id=f"{modality.value}-src",
        observed_at=observed_at,
        features=features or {"x": 1.0},
        provenance_hash="a" * 64,
    )


class MultimodalFuserTests(unittest.TestCase):
    def test_mandatory_present_is_decision_ready(self) -> None:
        fuser = MultimodalFuser(window_seconds=10.0)
        state = fuser.fuse(
            [make(Modality.NETWORK), make(Modality.POWER), make(Modality.SECURITY)],
            now=NOW,
        )
        self.assertTrue(state.decision_ready)
        self.assertEqual(state.missing_mandatory, ())

    def test_missing_mandatory_blocks_decision(self) -> None:
        fuser = MultimodalFuser(window_seconds=10.0)
        state = fuser.fuse([make(Modality.NETWORK), make(Modality.POWER)], now=NOW)
        self.assertFalse(state.decision_ready)
        self.assertIn("security", state.missing_mandatory)

    def test_stale_observation_is_dropped(self) -> None:
        fuser = MultimodalFuser(window_seconds=5.0)
        stale = make(Modality.SECURITY, observed_at=NOW - timedelta(seconds=30))
        state = fuser.fuse([make(Modality.NETWORK), make(Modality.POWER), stale], now=NOW)
        self.assertFalse(state.decision_ready)
        self.assertIn("security", state.stale_modalities)

    def test_provenance_hash_required(self) -> None:
        with self.assertRaises(ValueError):
            ModalityObservation.create(
                Modality.NETWORK,
                source_id="x",
                observed_at=NOW,
                features={"x": 1.0},
                provenance_hash="not-a-hash",
            )

    def test_nonfinite_feature_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make(Modality.NETWORK, features={"x": float("inf")})

    def test_freshest_observation_wins(self) -> None:
        fuser = MultimodalFuser(window_seconds=60.0)
        older = make(Modality.NETWORK, observed_at=NOW - timedelta(seconds=10), features={"rtt": 100.0})
        newer = make(Modality.NETWORK, observed_at=NOW, features={"rtt": 20.0})
        state = fuser.fuse([older, newer, make(Modality.POWER), make(Modality.SECURITY)], now=NOW)
        network = next(o for o in state.observations if o.modality is Modality.NETWORK)
        self.assertEqual(network.features["rtt"], 20.0)


if __name__ == "__main__":
    unittest.main()
