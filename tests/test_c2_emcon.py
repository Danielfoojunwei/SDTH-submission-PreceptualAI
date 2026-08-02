"""The command link, emission control, and the interaction between them.

CNPC is a protected, separately regulated traffic class — ITU-R sets aside
5030-5091 MHz and 960-977 MHz for it, RTCA DO-362/DO-377 give it its own MOPS,
FAA TSO-C213 covers its radios, and FCC Part 88 governs the band. Loss of it
triggers a pre-planned contingency under RTCA DO-400, not a routing decision.
See docs/DEFENCE_STANDARDS.md.

The tests that matter most here are the ones about EMCON: a commanded silence
and a jamming-induced link loss produce the same observation, and a vehicle
that runs its lost-link contingency because it was obeying an order has been
defeated by its own doctrine.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from kval_node.c2link import (
    CNPC_PROTECTED_BANDS,
    C2Budget,
    C2LinkError,
    C2LinkMonitor,
    C2State,
    ContingencyAction,
    DirectionalHealth,
    InteroperabilityDeclaration,
    LinkRole,
    protected_band_for,
    rank_within_roles,
    validate_c2_profile,
)
from kval_node.emcon import (
    EmconError,
    EmconOrder,
    EmconState,
    assess_emcon,
    is_more_restrictive,
)

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)

BUDGET = C2Budget(
    max_latency_ms=200.0,
    max_loss_percent=2.0,
    loss_confirmation_seconds=5.0,
    contingency_after_seconds=30.0,
    action=ContingencyAction.RETURN_TO_RECOVERY,
)

HEALTHY = DirectionalHealth(True, 50.0, 0.5, NOW)
SLOW = DirectionalHealth(True, 900.0, 0.5, NOW)
DARK = DirectionalHealth(False, None, None, None)


class ProtectedSpectrumTests(unittest.TestCase):
    def test_the_itu_allocations_are_the_published_ones(self) -> None:
        bands = {name: (low, high) for name, low, high in CNPC_PROTECTED_BANDS}
        self.assertEqual(bands["c-band-cnpc"], (5_030_000_000, 5_091_000_000))
        self.assertEqual(bands["l-band-cnpc"], (960_000_000, 977_000_000))

    def test_a_band_inside_the_allocation_is_recognised(self) -> None:
        self.assertEqual(protected_band_for(5_040_000_000, 5_050_000_000), "c-band-cnpc")

    def test_a_band_straddling_the_edge_is_not_protected(self) -> None:
        self.assertIsNone(protected_band_for(5_020_000_000, 5_050_000_000))

    def test_a_c2_profile_in_an_ism_band_is_flagged(self) -> None:
        # Not forbidden — military C2 uses allocations civil aviation never
        # touches — but a command link sharing 2.4 GHz with every consumer
        # device around has a very different jamming exposure.
        band, reasons = validate_c2_profile(
            profile_id="c2-wifi",
            role=LinkRole.C2_CNPC,
            freq_low_hz=2_400_000_000,
            freq_high_hz=2_483_500_000,
        )
        self.assertIsNone(band)
        self.assertIn("C2_LINK_IS_OUTSIDE_PROTECTED_CNPC_SPECTRUM_AND_SHARES_ITS_BAND", reasons)

    def test_a_c2_profile_in_protected_spectrum_is_named(self) -> None:
        band, reasons = validate_c2_profile(
            profile_id="c2-cband",
            role=LinkRole.C2_CNPC,
            freq_low_hz=5_040_000_000,
            freq_high_hz=5_050_000_000,
        )
        self.assertEqual(band, "c-band-cnpc")
        self.assertIn("C2_LINK_IS_IN_PROTECTED_CNPC_SPECTRUM:c-band-cnpc", reasons)

    def test_a_c2_profile_with_no_declared_spectrum_says_so(self) -> None:
        _band, reasons = validate_c2_profile(
            profile_id="c2", role=LinkRole.C2_CNPC, freq_low_hz=None, freq_high_hz=None
        )
        self.assertIn("C2_PROFILE_DECLARES_NO_SPECTRUM_SO_ITS_PROTECTION_IS_UNKNOWN", reasons)

    def test_a_payload_profile_is_not_checked(self) -> None:
        band, reasons = validate_c2_profile(
            profile_id="video", role=LinkRole.PAYLOAD, freq_low_hz=None, freq_high_hz=None
        )
        self.assertIsNone(band)
        self.assertEqual(reasons, ())


class RoleSegregationTests(unittest.TestCase):
    """The safety property: a payload link can never outrank a command link."""

    def test_a_high_scoring_payload_link_cannot_win_the_c2_slot(self) -> None:
        ranking = rank_within_roles(
            [
                ("c2-radio", LinkRole.C2_CNPC, 0.2),
                ("fat-video-downlink", LinkRole.PAYLOAD, 0.99),
            ]
        )
        self.assertEqual(ranking.selection_for(LinkRole.C2_CNPC), "c2-radio")
        self.assertEqual(ranking.selection_for(LinkRole.PAYLOAD), "fat-video-downlink")

    def test_ranking_within_a_role_still_prefers_the_better_link(self) -> None:
        ranking = rank_within_roles(
            [
                ("c2-poor", LinkRole.C2_CNPC, 0.2),
                ("c2-good", LinkRole.C2_CNPC, 0.8),
            ]
        )
        self.assertEqual(ranking.by_role["c2_cnpc"], ("c2-good", "c2-poor"))

    def test_no_c2_candidate_is_stated_not_filled_from_another_role(self) -> None:
        ranking = rank_within_roles([("video", LinkRole.PAYLOAD, 0.9)])
        self.assertIsNone(ranking.selection_for(LinkRole.C2_CNPC))
        self.assertIn(
            "NO_C2_CANDIDATE_OFFERED_SO_NO_COMMAND_LINK_CAN_BE_SELECTED", ranking.reasons
        )

    def test_the_segregation_is_stated_on_every_ranking(self) -> None:
        ranking = rank_within_roles([("c2", LinkRole.C2_CNPC, 0.5)])
        self.assertIn("CANDIDATES_RANKED_WITHIN_ROLE_NEVER_ACROSS_ROLES", ranking.reasons)

    def test_ties_break_deterministically(self) -> None:
        first = rank_within_roles([("b", LinkRole.PAYLOAD, 0.5), ("a", LinkRole.PAYLOAD, 0.5)])
        second = rank_within_roles([("a", LinkRole.PAYLOAD, 0.5), ("b", LinkRole.PAYLOAD, 0.5)])
        self.assertEqual(first.by_role, second.by_role)

    def test_ranking_serializes(self) -> None:
        json.dumps(rank_within_roles([("c2", LinkRole.C2_CNPC, 0.5)]).to_dict())


class BudgetTests(unittest.TestCase):
    def test_a_contingency_must_be_declared_in_advance(self) -> None:
        # RTCA DO-400 requires a pre-planned response. An undeclared
        # contingency is one chosen at the moment of failure.
        with self.assertRaises(C2LinkError):
            C2Budget.from_dict(
                {
                    "max_latency_ms": 200.0,
                    "max_loss_percent": 2.0,
                    "loss_confirmation_seconds": 5.0,
                    "contingency_after_seconds": 30.0,
                }
            )

    def test_a_contingency_cannot_precede_loss_confirmation(self) -> None:
        with self.assertRaises(C2LinkError):
            C2Budget(
                max_latency_ms=200.0,
                max_loss_percent=2.0,
                loss_confirmation_seconds=30.0,
                contingency_after_seconds=5.0,
                action=ContingencyAction.HOLD,
            )

    def test_the_budget_round_trips(self) -> None:
        self.assertEqual(C2Budget.from_dict(BUDGET.to_dict()), BUDGET)

    def test_a_link_outside_the_budget_is_not_within_it(self) -> None:
        self.assertFalse(SLOW.within(BUDGET))
        self.assertTrue(HEALTHY.within(BUDGET))

    def test_a_link_with_no_measurement_is_not_within_budget(self) -> None:
        # Absent evidence is not evidence of compliance.
        self.assertFalse(DirectionalHealth(True, None, None, NOW).within(BUDGET))


class LostLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = C2LinkMonitor(BUDGET)

    def _observe(self, seconds: float, command=DARK, ret=DARK, silence=False):
        return self.monitor.observe(
            command_path=command,
            return_path=ret,
            now=NOW + timedelta(seconds=seconds),
            emission_silence_commanded=silence,
        )

    def test_a_healthy_link_is_nominal(self) -> None:
        assessment = self._observe(0, command=HEALTHY, ret=HEALTHY)
        self.assertIs(assessment.state, C2State.NOMINAL)
        self.assertIsNone(assessment.contingency)

    def test_a_brief_interruption_does_not_commit_the_platform(self) -> None:
        # A contingency triggered on one bad sample is one an adversary fires
        # with a single pulse.
        self.assertIs(self._observe(0).state, C2State.DEGRADED)
        self.assertIs(self._observe(2).state, C2State.DEGRADED)

    def test_loss_is_confirmed_before_it_is_acted_on(self) -> None:
        self._observe(0)
        self.assertIs(self._observe(6).state, C2State.LOST)
        self.assertIsNone(self.monitor.declared_contingency())

    def test_the_pre_planned_contingency_is_declared_on_time(self) -> None:
        self._observe(0)
        assessment = self._observe(31)
        self.assertIs(assessment.state, C2State.CONTINGENCY)
        self.assertIs(assessment.contingency, ContingencyAction.RETURN_TO_RECOVERY)

    def test_the_node_never_chooses_the_contingency(self) -> None:
        # It executes what the operator declared, whatever the conditions.
        monitor = C2LinkMonitor(
            C2Budget(200.0, 2.0, 1.0, 2.0, ContingencyAction.TERMINATE)
        )
        monitor.observe(command_path=DARK, return_path=DARK, now=NOW)
        assessment = monitor.observe(
            command_path=DARK, return_path=DARK, now=NOW + timedelta(seconds=3)
        )
        self.assertIs(assessment.contingency, ContingencyAction.TERMINATE)

    def test_a_recovered_link_does_not_silently_unwind_a_contingency(self) -> None:
        # The platform is executing a plan. Cancelling it automatically would
        # leave it somewhere neither the plan nor the operator expects.
        self._observe(0)
        self._observe(31)
        recovered = self._observe(40, command=HEALTHY, ret=HEALTHY)
        self.assertIs(recovered.state, C2State.CONTINGENCY)
        self.assertIn(
            "CONTINGENCY_ALREADY_DECLARED_AND_ONLY_AN_OPERATOR_CLEARS_IT", recovered.reasons
        )

    def test_an_operator_can_clear_it(self) -> None:
        self._observe(0)
        self._observe(31)
        self.monitor.reset()
        self.assertIs(self._observe(50, command=HEALTHY, ret=HEALTHY).state, C2State.NOMINAL)

    def test_a_slow_link_is_degraded_even_though_it_is_present(self) -> None:
        # "Best of a bad set" is how a vehicle flies on a link that cannot
        # command it.
        self.assertIs(self._observe(0, command=SLOW, ret=HEALTHY).state, C2State.DEGRADED)

    def test_a_dark_return_path_alone_is_degraded_not_lost(self) -> None:
        assessment = self._observe(0, command=HEALTHY, ret=DARK)
        self.assertIs(assessment.state, C2State.DEGRADED)
        self.assertIn("RETURN_PATH_UNAVAILABLE_WHILE_COMMANDS_STILL_ARRIVE", assessment.reasons)

    def test_assessment_serializes(self) -> None:
        json.dumps(self._observe(0, command=HEALTHY, ret=HEALTHY).to_dict())


class EmconTests(unittest.TestCase):
    def test_no_order_means_unrestricted_not_silent(self) -> None:
        # Inventing silence from missing information would suppress the
        # lost-link contingency of every unconfigured platform.
        posture = assess_emcon(None, now=NOW)
        self.assertIs(posture.state, EmconState.UNRESTRICTED)
        self.assertFalse(posture.silence_commanded())

    def test_an_order_must_expire(self) -> None:
        # An order without an expiry survives the loss of whoever issued it.
        with self.assertRaises(EmconError):
            EmconOrder(EmconState.SILENT, NOW, NOW, "task-force")

    def test_each_level_suppresses_more_than_the_last(self) -> None:
        levels = [
            EmconState.UNRESTRICTED,
            EmconState.RESTRICTED,
            EmconState.MISSION_CRITICAL_ONLY,
            EmconState.SILENT,
        ]
        counts = [
            len(
                assess_emcon(
                    EmconOrder(level, NOW, NOW + timedelta(hours=1), "hq"), now=NOW
                ).suppressed_roles
            )
            for level in levels
        ]
        self.assertEqual(counts, sorted(counts))

    def test_restricted_suppresses_only_housekeeping(self) -> None:
        posture = assess_emcon(
            EmconOrder(EmconState.RESTRICTED, NOW, NOW + timedelta(hours=1), "hq"), now=NOW
        )
        self.assertEqual(posture.suppressed_roles, ("bulk",))
        self.assertTrue(posture.permits("c2_cnpc"))
        self.assertTrue(posture.permits("payload"))

    def test_silence_still_permits_receive_only_sensing(self) -> None:
        posture = assess_emcon(
            EmconOrder(EmconState.SILENT, NOW, NOW + timedelta(hours=1), "hq"), now=NOW
        )
        self.assertIn(
            "RECEIVE_ONLY_SENSING_CONTINUES_BECAUSE_RECEIVING_EMITS_NOTHING", posture.reasons
        )

    def test_an_expired_order_stops_suppressing(self) -> None:
        # Otherwise an adversary who jams the lifting order keeps the
        # contingency suppressed indefinitely.
        order = EmconOrder(EmconState.SILENT, NOW, NOW + timedelta(minutes=10), "hq")
        posture = assess_emcon(order, now=NOW + timedelta(hours=2))
        self.assertTrue(posture.expired)
        self.assertFalse(posture.silence_commanded())
        self.assertEqual(posture.suppressed_roles, ())
        self.assertIn(
            "SILENCE_PAST_THE_ORDERED_WINDOW_IS_NOT_AN_ORDER_BEING_OBEYED", posture.reasons
        )

    def test_an_imminent_expiry_is_announced(self) -> None:
        order = EmconOrder(EmconState.SILENT, NOW, NOW + timedelta(seconds=30), "hq")
        posture = assess_emcon(order, now=NOW)
        self.assertIn("EMCON_ORDER_EXPIRES_SHORTLY_AND_SUPPRESSION_WILL_LIFT", posture.reasons)

    def test_ordering_between_states(self) -> None:
        self.assertTrue(is_more_restrictive(EmconState.SILENT, EmconState.RESTRICTED))
        self.assertFalse(is_more_restrictive(EmconState.RESTRICTED, EmconState.SILENT))

    def test_an_unknown_state_is_refused(self) -> None:
        with self.assertRaises(EmconError):
            EmconOrder.from_dict(
                {"state": "invisible", "issued_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(hours=1)).isoformat(), "authority": "hq"}
            )

    def test_order_round_trips(self) -> None:
        order = EmconOrder(EmconState.SILENT, NOW, NOW + timedelta(hours=1), "hq")
        self.assertEqual(EmconOrder.from_dict(order.to_dict()), order)

    def test_posture_serializes(self) -> None:
        json.dumps(assess_emcon(None, now=NOW).to_dict())


class EmconAndLostLinkTests(unittest.TestCase):
    """The interaction this whole design exists for.

    A commanded silence and a jamming-induced link loss produce the same
    observation. A vehicle that runs its contingency because it was obeying an
    order has been defeated without the adversary doing anything.
    """

    def _silence(self, minutes: int = 30) -> bool:
        return assess_emcon(
            EmconOrder(EmconState.SILENT, NOW, NOW + timedelta(minutes=minutes), "task-force"),
            now=NOW,
        ).silence_commanded()

    def test_jamming_both_directions_declares_the_contingency(self) -> None:
        monitor = C2LinkMonitor(BUDGET)
        monitor.observe(command_path=DARK, return_path=DARK, now=NOW)
        assessment = monitor.observe(
            command_path=DARK, return_path=DARK, now=NOW + timedelta(seconds=31)
        )
        self.assertIs(assessment.state, C2State.CONTINGENCY)

    def test_commanded_silence_does_not_declare_a_contingency(self) -> None:
        # The vehicle is obeying an order. Commands still arrive; only the
        # return path is dark, and that is compliance.
        monitor = C2LinkMonitor(BUDGET)
        for seconds in (0, 31, 600):
            assessment = monitor.observe(
                command_path=HEALTHY,
                return_path=DARK,
                now=NOW + timedelta(seconds=seconds),
                emission_silence_commanded=self._silence(),
            )
            self.assertIs(assessment.state, C2State.NOMINAL)
            self.assertIsNone(assessment.contingency)

    def test_the_suppression_is_recorded_not_silent(self) -> None:
        monitor = C2LinkMonitor(BUDGET)
        assessment = monitor.observe(
            command_path=HEALTHY,
            return_path=DARK,
            now=NOW,
            emission_silence_commanded=self._silence(),
        )
        self.assertTrue(assessment.emcon_suppressed_return_path)
        self.assertIn("EMCON_SILENCE_IS_NOT_EVIDENCE_OF_A_LOST_LINK", assessment.reasons)

    def test_emcon_never_suppresses_loss_of_the_command_path(self) -> None:
        # The asymmetry that makes this safe: nothing about a silence order
        # stops an operator transmitting *to* the vehicle. A dark command path
        # under EMCON is still a lost link.
        monitor = C2LinkMonitor(BUDGET)
        monitor.observe(
            command_path=DARK, return_path=DARK, now=NOW,
            emission_silence_commanded=self._silence(),
        )
        assessment = monitor.observe(
            command_path=DARK,
            return_path=DARK,
            now=NOW + timedelta(seconds=31),
            emission_silence_commanded=self._silence(),
        )
        self.assertIs(assessment.state, C2State.CONTINGENCY)

    def test_jamming_during_an_emcon_window_still_reaches_the_contingency(self) -> None:
        # The attack this defends: jam during a known silence window and hope
        # the suppression covers for you. It covers the return path only.
        monitor = C2LinkMonitor(BUDGET)
        silence = self._silence()
        monitor.observe(
            command_path=DARK, return_path=DARK, now=NOW, emission_silence_commanded=silence
        )
        assessment = monitor.observe(
            command_path=DARK,
            return_path=DARK,
            now=NOW + timedelta(seconds=40),
            emission_silence_commanded=silence,
        )
        self.assertIs(assessment.contingency, ContingencyAction.RETURN_TO_RECOVERY)

    def test_suppression_lifts_with_the_order_even_if_silence_persists(self) -> None:
        # An adversary who jams the lifting order must not get indefinite
        # suppression.
        expired = assess_emcon(
            EmconOrder(EmconState.SILENT, NOW, NOW + timedelta(minutes=1), "hq"),
            now=NOW + timedelta(hours=1),
        )
        self.assertFalse(expired.silence_commanded())


class InteroperabilityTests(unittest.TestCase):
    def test_the_claimed_loi_is_modest_and_explained(self) -> None:
        # STANAG 4586 LOI 4 and 5 mean controlling the vehicle. A comms-layer
        # component gives a control station none of that, and claiming it is
        # the sort of overclaim found during integration rather than a pitch.
        declaration = InteroperabilityDeclaration().to_dict()
        self.assertEqual(declaration["stanag_4586_loi"], 1)
        self.assertFalse(declaration["controls_vehicle"])
        self.assertFalse(declaration["controls_payload"])

    def test_every_domain_is_covered_because_link_selection_is_agnostic(self) -> None:
        declaration = InteroperabilityDeclaration().to_dict()
        self.assertEqual(
            set(declaration["domains"]), {"land", "sea", "air", "subsurface"}
        )

    def test_declaration_serializes(self) -> None:
        json.dumps(InteroperabilityDeclaration().to_dict())


if __name__ == "__main__":
    unittest.main()
