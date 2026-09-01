"""STEP 5 - DELIVER. Turn the chosen split into what ships and what it is worth: the backup
catch-all blend, pool compression, the connector-pool config files, and the impact frames.

The sN_ prefix is the order the PIPELINE runs, as a reading aid - not a claim that a
module is used only in that step (band_projection, for one, is read by delivery too).
Imports inside the package are ABSOLUTE (routing_optimiser.sN_x.y) on purpose: relative
dots would have to change every time a file moves between steps, which is exactly the
edit that goes wrong silently.
"""
