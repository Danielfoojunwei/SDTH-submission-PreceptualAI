from __future__ import annotations

import unittest

from kval_node.telemetry import parse_ping_output


class TelemetryParserTests(unittest.TestCase):
    def test_parses_linux_iputils_summary(self) -> None:
        output = """
3 packets transmitted, 3 received, 0% packet loss, time 2002ms
rtt min/avg/max/mdev = 11.100/12.200/13.300/0.900 ms
"""
        result = parse_ping_output(output)
        self.assertEqual(result.loss_percent, 0.0)
        self.assertEqual(result.rtt_ms, 12.2)
        self.assertEqual(result.jitter_ms, 0.9)

    def test_records_total_loss_without_inventing_latency(self) -> None:
        result = parse_ping_output("3 packets transmitted, 0 received, 100% packet loss, time 2030ms")
        self.assertEqual(result.loss_percent, 100.0)
        self.assertIsNone(result.rtt_ms)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()

