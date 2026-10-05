"""V2 Phase 9 — ML-based LST reduction prediction.

Phase 9 is intentionally separate from frozen Phases 2-8.  It learns an
empirical supervised regression target from observed W4 multi-year changes:

    delta_lst_C = LST(next year, W4) - LST(current year, W4)

At scenario time, proposed tree densities are converted to an explicit
assumed vegetation-cover change parameter before model inference.  The output
is therefore a predicted LST reduction under an assumed vegetation-change
scenario, not proof of causal tree-cooling.
"""
