# Transaction Routing Optimiser

Batch tool that decides **what fraction of each transaction cell to send to each
gateway/MID**, to maximise the success (authorisation) rate while staying inside
your risk / chargeback (Visa VAMP) limits. It plugs into your existing VAMP
forecast pipeline: it consumes the "pre" (baseline) forecast and produces a
proposed split, an impact dashboard, compressed rules, and ready-to-ship JSON
routing configs.

A **cell** = one `RPGT (transaction type) x Currency x Bank` combination.

## The flow

```
Pre-forecast (baseline)
   -> Split engine   [dropdown of 4 methods + conversion<->risk slider]
   -> Normalised split table   (profile key -> gateway %, the shared output)
   -> Impact dashboard         (risk + success-rate / revenue charts)
   -> Volume-weighted k-means  (compress the config count)
   -> JSON config generator
   -> Export outputs           (splits, configs, charts, summary)
```

The key design point: **every engine reads the same input and writes the same
output** (a table of gateway percentages per cell), so you can switch methods
from a dropdown and nothing downstream changes.

## The four engines (the dropdown)

Every engine decides the same thing — how to divide each cell's payments between
its gateways — and they all read the same inputs and produce the same kind of
output, so you can switch between them from the dropdown. The **conversion ↔ risk
slider** works with all of them: slide toward `1.0` to chase the most approved
payments, toward `0.0` to play it safest.

Think of each gateway as a different door you can send a payment through. Some
doors let more payments succeed; some are riskier (more chargebacks). The engine
decides how many payments go through each door.

### Genetic algorithm — the default (what we run in production)

- Tries out loads of different ways to split the traffic between the doors.
- Keeps the ones that work best, then mixes and tweaks them and tries again — a bit like breeding for the strongest result over many rounds.
- Best at obeying *all* the rules at once (risk caps, per-bank targets), which is why it's the default.

### Softmax allocation

- Sends more payments to the doors that are better at getting approved.
- But never puts everything on one door — it always keeps some spread.
- Like picking your strongest players but not benching everyone else.

### Thompson (bandit)

- Sends most traffic to whichever door is *probably* the best right now.
- Keeps giving the newer / less-tested doors a few tries, in case one turns out great.
- Like mostly playing your favourite game but trying new ones too, so you don't miss a hidden winner.

### Portfolio (mean-CVaR)

- Works like spreading your pocket money across different piggy banks.
- Picks good performers but avoids the unpredictable ones that could suddenly go bad (a chargeback spike).
- Gives up a tiny bit of success rate in exchange for fewer nasty surprises.

## Constraints

**Hard** (a split is only valid if all hold): max share per gateway, per-cell
VAMP cap `sum(share x risk) <= cap`, banned/forced gateways.
**Soft** (penalised, not forced): exploration floor, stability, gateway
preferences.

## First-time setup

Tested on macOS with **Python 3.8+** (also runs on Linux). Steps 1–3 get the UI
running against previously-run outputs; steps 4–5 are only needed to pull fresh
data from BigQuery.

### 1. Get the code

```bash
git clone <YOUR_REPO_URL> routing_optimiser
cd routing_optimiser
```

### 2. Create a Python environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the UI

```bash
streamlit run app/streamlit_app.py
```

Opens at <http://localhost:8501>. You can use Tabs 3–6 (routing engine, impact,
k-means compression, config generator) **without BigQuery** — on the Forecast
tab choose **"Load a previously-run baseline"** and point it at a folder under
`data/outputs/<MONTH>/<COMPANY>/`. (These output folders are gitignored, so on a
fresh clone you'll need to run the pipeline once — steps 4–5 — or copy an
existing outputs folder in.)

### 4. Install the Google Cloud SDK (only for live BigQuery runs)

The forecast and attempts extracts read from BigQuery (project
`sapient-tangent-172609`), so a **live** run needs the `gcloud` CLI. The SDK is
**not** committed to this repo (it's gitignored) — install your own:

```bash
# macOS (Homebrew)
brew install --cask google-cloud-sdk

# …or the official installer (macOS / Linux)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL                       # reload your shell so `gcloud` is on PATH

gcloud --version                     # verify it's installed
```

### 5. Authenticate to BigQuery

```bash
gcloud auth login                                                # your Google account
gcloud auth application-default login                            # creds the Python client uses
gcloud config set project sapient-tangent-172609
gcloud auth application-default set-quota-project sapient-tangent-172609
```

The Python BigQuery client uses the **Application Default Credentials** created
by `gcloud auth application-default login`. You need access to the
`sapient-tangent-172609` project — if a query returns a 403, ask your admin to
grant BigQuery access.

A live forecast now works: on the Forecast tab pick **"Run VAMP pipeline"**, or
run it headless (writes to `data/outputs/<MONTH>/<COMPANY>/`):

```bash
python main.py
```

Extracts are cached to parquet under `data/cache/`, so subsequent runs only
re-query BigQuery on a cache miss (e.g. a new month or company).

### 6. After changing any backend code

Streamlit reuses compiled bytecode, and stale `.pyc` files are the most common
cause of "it still looks wrong after my edit". Clear the cache and fully restart
(Ctrl+C the process — closing the browser tab is not enough):

```bash
find . -name __pycache__ -type d -exec rm -rf {} +
streamlit run app/streamlit_app.py
```

### Optional: sanity test the engines

```bash
python scripts/test_engines.py
```

## Wiring in your real data

The real VAMP forecast pipeline is vendored under `src/vamp_pipeline/` (your
`DataExtractor` → `ActuarialEngine` → `AllocationEngine` → `ExportManager`), and
its extract queries live in `queries/`. The gateway→MID mapping
(`data/mappings/Master_MID_List.csv`) is wired as the pipeline's `mid_list_file`,
so `vampMid`s resolve exactly as in production. Pick the baseline source on the
Forecast tab:

1. **Run VAMP pipeline.** Runs the full pipeline from the settings you set
   (mapped to the pipeline's `settings.yaml` schema). It **reuses the cached
   BigQuery extracts** in `data/cache/{month_var}/{company}/` automatically, so
   you can **regenerate a new forecast from cached inputs** — change targets,
   overrides, split rules or actuarial settings and re-run; it only re-queries
   BigQuery on a cache miss (e.g. a new month 0 or company). The "Reuse cached
   actuarial curves" toggle maps to `load_curves_from_cache`.
2. **Load a previously-run baseline.** Point at a prior pipeline output folder
   (or its `effective_rate_impact.csv`). No BigQuery, no new forecast computed.
3. **Synthesise from attempts (offline).** Stand-in baseline for the sample.

The mapping is: the pipeline's `Sim_Sales` → cell `volume`, `Sim_Rate` (VAMPs /
sales per gateway) → `risk_rate`, and each gateway's share of the cell →
`baseline_share`, at the pipeline's `vampMid × rpgt × BIN × Currency` grain.
Success rates come from the attempts extract (`sql/attempts_success.sql`) with
empirical-Bayes shrinkage.

Run the pipeline headless and inspect the baseline:

```bash
python scripts/run_forecast_pipeline.py --settings config/settings.yaml   # live (needs BigQuery)
python scripts/run_forecast_pipeline.py --pre data/outputs/MAY/TotalAV/   # from prior outputs
```

## Honest caveats

- The routing optimiser now uses the **real pipeline baseline** when you run it
  live or point at its outputs; the synthesiser is only a no-BigQuery fallback.
- The per-cell VAMP cap is enforced cell-by-cell; **global per-acquirer caps**
  need an aggregation layer on top.
- On the bundled 50-row sample the attempts data and a real pipeline `pre` won't
  share the same BIN/gateway space, so success rates fall back to the pooled
  mean. On your real data (same BIN/gateway keys) they join properly.

## Input config files

`config/inputs/` holds editable JSON the Forecast tab loads by default (or you
can upload your own): `test_gateways.json`, `thermometer_config.json`,
`gateway_volume_overrides.json`.

## Layout

```
src/routing_optimiser/
  schema.py            column contracts + cell/profile keys
  constraints.py       HardConstraints / SoftConstraints / OptimiserSettings
  success_rates.py     empirical-Bayes per-cell gateway success rates
  sql_runner.py        run .sql extracts against BigQuery, cache to parquet
  forecast_pipeline.py adapter: UI settings -> pipeline config; run it; read pre
  data_loader.py       load forecast (real pipeline pre or synthesised) -> cells
  engines/             base + engines (softmax, thompson, portfolio; genetic_ref reference) + registry
                       (genetic is dispatched separately via genetic_global)
  optimiser.py         run an engine across all cells; slider sweep
  impact.py            revenue uplift, key contributors, gateway volume shift
  kmeans_compress.py   volume-weighted k-means compression
  config_generator.py  JSON routing configs
src/vamp_pipeline/     the real forecast pipeline (DataExtractor, ActuarialEngine,
                       AllocationEngine, ExportManager, utils)
queries/               the pipeline's BigQuery extracts (fcast_query.sql, etc.)
sql/attempts_success.sql   attempts/success extract for success rates
app/streamlit_app.py   the UI (5 tabs)
scripts/               run_pipeline.py, run_forecast_pipeline.py, test_engines.py
config/settings.yaml   VAMP pipeline settings (the UI mirrors this)
config/inputs/         test_gateways / thermometer / gateway_volume_overrides JSON
```
