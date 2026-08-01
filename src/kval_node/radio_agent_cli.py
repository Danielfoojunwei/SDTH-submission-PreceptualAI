from __future__ import annotations

import argparse

from .radio_agent import RadioAgent, RadioAgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="kval-radio-agent")
    parser.add_argument("--config", default="/etc/kval/radio-agent.json")
    args = parser.parse_args()
    RadioAgent(RadioAgentConfig.load(args.config)).serve_forever()


if __name__ == "__main__":
    main()
