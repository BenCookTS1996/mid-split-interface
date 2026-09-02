"""STEP 1 - EXTRACT. Get the raw material: run the SQL, load it, turn attempts into
per-profile success rates, and hold the column contract everything else relies on.

The sN_ prefix is the order the PIPELINE runs, as a reading aid - not a claim that a
module is used only in that step (band_projection, for one, is read by delivery too).
Imports inside the package are ABSOLUTE (routing_optimiser.sN_x.y) on purpose: relative
dots would have to change every time a file moves between steps, which is exactly the
edit that goes wrong silently.
"""
