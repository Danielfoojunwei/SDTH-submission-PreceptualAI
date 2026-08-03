"""Validation harness. **Not part of the node, and never importable from it.**

Everything under ``sim/`` fabricates link measurements. That is exactly what
``src/`` is forbidden to do — CI greps for simulation providers there, and the
whole product claim is that a reported measurement was measured. So the
boundary is absolute and mechanically enforced:

* ``sim`` imports ``kval_node``. ``kval_node`` must never import ``sim``.
* ``scripts/verify_integration.py`` fails the build if that reverses.
* Nothing here produces a signed artefact a node would accept as evidence.

**What a result from this harness is.** A statement about the *logic*: given
these measurements, the allocator placed these flows this way and refused those.
It is not a statement about radio. No number out of here is evidence about RF
performance, fade depth, or how any real vehicle behaves — the propagation used
is a coarse textbook model chosen to exercise decision paths, not to be right.
"""
