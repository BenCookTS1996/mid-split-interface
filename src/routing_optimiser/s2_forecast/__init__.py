"""STEP 2 - FORECAST. Run the four-phase pipeline that produces the BASELINE: how much
volume and how much fraud each profile is expected to carry. One adapter per card scheme.

The sN_ prefix is the order the PIPELINE runs, as a reading aid - not a claim that a
module is used only in that step (band_projection, for one, is read by delivery too).
Imports inside the package are ABSOLUTE (routing_optimiser.sN_x.y) on purpose: relative
dots would have to change every time a file moves between steps, which is exactly the
edit that goes wrong silently.
"""
