"""STEP 4 - SEARCH. Choose the split. The full-matrix GA is the engine; seed_search builds
the warm-start seeds and holds the reference fitness; band_projection scores the bands
exactly; numba_kernels and rowpar are the fast paths, each verified against a reference.

The sN_ prefix is the order the PIPELINE runs, as a reading aid - not a claim that a
module is used only in that step (band_projection, for one, is read by delivery too).
Imports inside the package are ABSOLUTE (routing_optimiser.sN_x.y) on purpose: relative
dots would have to change every time a file moves between steps, which is exactly the
edit that goes wrong silently.
"""
