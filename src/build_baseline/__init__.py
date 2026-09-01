"""BUILD BASELINE — the two vendored forecast pipelines, one per card scheme.

These produce the BASELINE the whole optimiser is measured against: how much volume and
how much fraud each cell is expected to carry over months 0-5. Four phases each:

    DataExtractor      pull from BigQuery (cached under
                       data/build_baseline_cached_input_data/)
    ActuarialEngine    fit the fraud curves
    AllocationEngine   route the forecast through the CURRENT split
    ExportManager      write the export CSVs, incl. vamp_t_period_prorata_export.csv

They are VENDORED - lifted from the original standalone repos and kept close to their
original shape on purpose, so a fix upstream can still be read across. That is why they
sit in their own folder rather than inside routing_optimiser: nothing in here is ours to
restructure freely.

Reached through the adapters in routing_optimiser/s2_forecast/, never directly by the
app - `run_vamp_pipeline` / `run_mastercard_pipeline` are the only supported entry points.
"""
