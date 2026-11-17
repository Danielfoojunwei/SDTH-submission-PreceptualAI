"""WirelessJEV Field Platform: one local web app for real SmartCar testing.

    .venv/bin/python -m wirelessjev.platform            # then open http://127.0.0.1:8770

It drives the same tested commands as the runbook (`wirelessjev.onecar`,
`wirelessjev.hil.*`, `wirelessjev.native_ext.coverage`) as subprocesses and
shows their output, the live lap, the site setup and the analysis in one page,
with an assistant that says what to do next.

Safety model (unchanged from the command line):
- read-only steps run from the page;
- steps that change the car's radio (`switch`, `roam`) are never run from the
  page: it shows the exact command to run in a terminal, where it asks for
  the typed confirmation;
- one car command at a time; the server listens on 127.0.0.1 only and refuses
  requests from other web pages;
- the SSH password comes from `WJ_SSH_PASSWORD` in the terminal that started
  the platform, never from the page, and is never written anywhere.
"""
