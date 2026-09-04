"""audit.py -- the leakage trap.

R2: labels are never features.  Nothing derived from the as-built strap map,
the fine-grid solve, or the signoff IR map may enter feature extraction.

This module enforces that at runtime rather than by inspection.  Inside
`leakage_trap()` every label-side accessor is replaced by one that raises
`LeakageError` naming the offending attribute, so a leak is a crash with a
stack trace pointing at the guilty line -- not a suspiciously good CV score.

Trapped surfaces
----------------
  Design.irmap                    the signoff IR map (the label itself)
  design._build_layout            builds the as-built strap map
  design._ground_truth_current    sub-tile concentration, label-side only
  design._reference_drop          a fine-grid solve on as-built straps

Writes are still permitted, so `load_design()` works inside the trap; only
READS raise.  That is deliberate: feature extraction must be able to load a
design, it just must not look at its labels.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List

from prism import design as _design_mod
from prism import io_csv as _io_mod


class LeakageError(RuntimeError):
    """Raised when feature extraction touches a label-side value."""


# Attribute name used to stash a trapped value so __init__ still works.
_STASH = "_trapped_irmap"

_TRAPPED_FUNCTIONS = [
    "_build_layout",
    "_ground_truth_current",
    "_reference_drop",
]


def _make_trapped_property() -> property:
    """A data descriptor that accepts writes but raises on read.

    `property` defines __set__, so it takes precedence over the instance
    __dict__ -- which is what lets it intercept a dataclass field.
    """

    def _get(self):
        raise LeakageError(
            "LEAKAGE: read of Design.irmap during feature extraction. "
            "irmap.csv is the signoff IR map -- it is the LABEL. Features may "
            "only use design_stats, modules, macros, instances, bumps, "
            "strap_planned and activity. If you need the label, compute it "
            "outside leakage_trap() with features.add_labels()."
        )

    def _set(self, value):
        object.__setattr__(self, _STASH, value)

    return property(_get, _set)


def _make_trapped_function(name: str):
    def _trapped(*_args, **_kwargs):
        raise LeakageError(
            f"LEAKAGE: call to design.{name}() during feature extraction. "
            f"That function sees the as-built strap map and/or the fine-grid "
            f"ground-truth solve. The floorplan-stage estimate must come from "
            f"solver.coarse_solver_from_design() and design.scenario_currents(), "
            f"which read published CSVs only."
        )

    _trapped.__name__ = name
    return _trapped


@contextlib.contextmanager
def leakage_trap() -> Iterator[List[str]]:
    """Trap every label-side accessor for the duration of the block.

    Yields the list of trapped surface names, so a caller can report what was
    guarded rather than merely asserting that something was.
    """
    trapped: List[str] = ["Design.irmap"] + [f"design.{n}" for n in _TRAPPED_FUNCTIONS]

    original_irmap = _io_mod.Design.__dict__.get("irmap", None)
    originals = {n: getattr(_design_mod, n) for n in _TRAPPED_FUNCTIONS}

    _io_mod.Design.irmap = _make_trapped_property()
    for n in _TRAPPED_FUNCTIONS:
        setattr(_design_mod, n, _make_trapped_function(n))

    try:
        yield trapped
    finally:
        if original_irmap is None:
            # dataclass field: no class attribute existed before the patch
            try:
                del _io_mod.Design.irmap
            except AttributeError:
                pass
        else:
            _io_mod.Design.irmap = original_irmap
        for n, fn in originals.items():
            setattr(_design_mod, n, fn)

        # Move any value written through the trapped setter back into the
        # instance dict is impossible from here (we hold no instances), so
        # restore access by making the stashed value visible again: the
        # dataclass field lookup falls back to __dict__ once the descriptor
        # is gone.  Instances written to during the trap expose the value
        # under _STASH; unstash on demand.


def unstash_irmap(design) -> None:
    """Restore `design.irmap` for an object that was constructed inside the
    trap (its __init__ write went to the stash instead of __dict__)."""
    if not hasattr(design, "irmap") or design.__dict__.get("irmap", None) is None:
        stashed = design.__dict__.get(_STASH, None)
        if stashed is not None:
            object.__setattr__(design, "irmap", stashed)


def assert_no_leakage(fn, *args, **kwargs):
    """Run `fn` under the trap and return its result.

    Any label-side access inside raises LeakageError.
    """
    with leakage_trap():
        return fn(*args, **kwargs)


if __name__ == "__main__":
    # Self-test: prove the trap actually fires.
    from prism.io_csv import load_design

    print("audit.py self-test")
    with leakage_trap() as guarded:
        print(f"  trapped surfaces: {', '.join(guarded)}")
        d = load_design("data/synthetic/syn_000")
        print("  load_design inside trap: OK (writes permitted)")
        for probe, call in [
            ("Design.irmap", lambda: d.irmap),
            ("design._build_layout", lambda: _design_mod._build_layout(0, {})),
            ("design._ground_truth_current",
             lambda: _design_mod._ground_truth_current({}, "idle", {})),
            ("design._reference_drop", lambda: _design_mod._reference_drop({}, {})),
        ]:
            try:
                call()
            except LeakageError:
                print(f"  read of {probe}: LeakageError raised as expected")
            else:
                raise SystemExit(f"TRAP FAILED: {probe} did not raise")
    unstash_irmap(d)
    print(f"  after trap, irmap readable again: {len(d.irmap):,} rows")
    print("LEAKAGE TRAP SELF-TEST: PASS")
