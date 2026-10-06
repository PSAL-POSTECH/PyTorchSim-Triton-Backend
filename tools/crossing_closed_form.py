"""What a CROSSING costs, said twice, so the cheap saying can be checked.

`_get_crossing_passes` answers how many passes of the unit a correspondence
takes by walking the pairs -- ONE ENTRY PER ELEMENT of the tile. Its answer does
not need them: swept over 19,800 quadruples, the answer is a function of five
shape-level numbers alone, with no two quadruples sharing those five and
disagreeing. This tool is where that second saying lives, and it does not run in
the compiler -- it exists so that the day the element walk is replaced, the
replacement has something to fail against.

WHAT IS PROVEN AND WHAT IS FITTED. The two are not the same and the difference
is the whole reason this file is a check rather than an implementation.

  Ws * Wd == numel          GEOMETRY. `depth_s` takes numel/Ws values and
                            `lane_d` takes Wd; a crossing maps one onto the
                            other, so the two counts must agree.

  passes == numel/(Ws*Wd)   GEOMETRY. crossing_passes cuts `deep = numel/Ws`
                            source rows into groups of `rows = Wd`, one pass
                            each. Checked: 11,946 crossings, 0 disagreements.

  the one-pass predicate    THREE BRANCHES DERIVED, ONE FITTED. Writing a flat
                            index as f = outer*(Ws*Is) + lane*Is + low gives the
                            Ws == 1, Is == 1 and Os == 1 cases by algebra. The
                            Wd == 1 branch is NOT derived -- it was written to
                            cover 168 counter-examples the other three missed,
                            and `Is == 1 or Os == 1` there is a fit. It holds
                            on both dim sets, which is evidence and not proof.

  is it a crossing at all   NOT DERIVED, except for the one-pass sub-case above.
                            The general predicate is whatever crossing_passes
                            accepts, and this file does not restate it.

So a green run here means the two sayings agree ON THE SWEPT SPACE. It is not a
proof, and the fitted branch is the first place to look when it goes red.

    python tools/crossing_closed_form.py [--wide]
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: TWO DISJOINT DIM SETS, because a rule fitted on powers of two would pass on
#: powers of two. `--wide` picks the second: 19,800 quadruples agree on the
#: first and 28,329 on the second, which is what "swept space" means here.
LANES = 256
DIMS = (1, 2, 4, 8, 16)
WIDE_DIMS = (1, 2, 3, 5, 6, 7, 12)


def _get_bank_parts(shape, axis):
    """`(lane extent, what walks inside it, what walks outside it)` -- the three
    numbers a flat index splits into around `axis`, and all this file reads."""
    return (shape[axis], math.prod(shape[axis + 1:]), math.prod(shape[:axis]))


def _get_closed_passes(s_shape, s_axis, d_shape, d_axis):
    """How many crossing passes the shapes alone say, or None for no crossing.
    THE COUNT IS GEOMETRY AND THE PREDICATE IS NOT -- see the module docstring;
    only the one-pass case is stated here, so None means "not one pass"."""
    numel = math.prod(s_shape)
    ws, is_, os_ = _get_bank_parts(s_shape, s_axis)
    wd, id_, _od = _get_bank_parts(d_shape, d_axis)
    if ws * wd != numel:
        return None
    if ws == 1:
        one = wd == numel and id_ == 1
    elif wd == 1:
        one = is_ == 1 or os_ == 1
    elif is_ == 1:
        one = id_ == ws
    elif os_ == 1:
        one = id_ == 1 and wd == is_
    else:
        one = False
    return numel // (ws * wd) if one else None


def _enumerate_shapes(rank, dims):
    """Every shape of `rank` from `dims` under 512 elements, with every axis."""
    for shape in itertools.product(dims, repeat=rank):
        if math.prod(shape) <= 512:
            for axis in range(rank):
                yield list(shape), axis


def _check_one_pass(dims, ranks):
    """(agreements, disagreements, the first few disagreeing quadruples)."""
    from pytorchsim_triton_compiler.elf.passes.lib_lanes import wanted_pairs
    same, differ, bad = 0, 0, []
    for rs in ranks:
        for ss, sa in _enumerate_shapes(rs, dims):
            for rd in ranks:
                for ds, da in _enumerate_shapes(rd, dims):
                    if math.prod(ss) != math.prod(ds):
                        continue
                    pairs = wanted_pairs(ss, sa, ds, da)
                    if pairs is None:
                        continue
                    walked = all(t == (f[1], f[0]) for f, t in pairs)
                    said = _get_closed_passes(ss, sa, ds, da) == 1
                    if walked == said:
                        same += 1
                    else:
                        differ += 1
                        if len(bad) < 8:
                            bad.append((ss, sa, ds, da, walked, said))
    return same, differ, bad


def _check_the_count(dims, ranks):
    """The same, for `passes == numel/(Ws*Wd)` over every crossing there is."""
    from pytorchsim_triton_compiler.elf.passes.lib_lanes import wanted_pairs
    from pytorchsim_triton_compiler.elf.passes.p08_fit_layout_to_lanes import (
        _get_crossing_passes)
    same, differ, bad = 0, 0, []
    for rs in ranks:
        for ss, sa in _enumerate_shapes(rs, dims):
            for rd in ranks:
                for ds, da in _enumerate_shapes(rd, dims):
                    if math.prod(ss) != math.prod(ds):
                        continue
                    pairs = wanted_pairs(ss, sa, ds, da)
                    if pairs is None:
                        continue
                    walked = _get_crossing_passes(pairs, ss, sa, LANES)
                    if walked is None:
                        continue
                    ws, wd = ss[sa], ds[da]
                    said = math.prod(ss) // (ws * wd) if ws * wd else None
                    if walked == said:
                        same += 1
                    else:
                        differ += 1
                        if len(bad) < 8:
                            bad.append((ss, sa, ds, da, walked, said))
    return same, differ, bad


def main():
    dims = WIDE_DIMS if "--wide" in sys.argv else DIMS
    ranks = (1, 2, 3)
    print(f"dims={dims} ranks={ranks}\n")
    rc = 0
    for name, check in (("passes == numel/(Ws*Wd)", _check_the_count),
                        ("the one-pass predicate", _check_one_pass)):
        same, differ, bad = check(dims, ranks)
        print(f"{name:26s} agree={same:6d}  differ={differ}")
        for ss, sa, ds, da, walked, said in bad:
            print(f"    {ss}a{sa} -> {ds}a{da}   walked={walked}  said={said}")
        rc |= 1 if differ else 0
    print("\nagreement on the swept space is not a proof; the fitted branch is "
          "`Wd == 1`" if not rc else "\nDISAGREEMENT -- see the module docstring")
    return rc


if __name__ == "__main__":
    sys.exit(main())
