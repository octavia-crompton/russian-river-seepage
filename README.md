# Russian River Seepage

Mass-balance workflow for the Russian River estuary.
This repository merges hydrologic and oceanographic records 
(USGS, NOAA CO-OPS, visitor-center gauges, inlet-state logs, and a stage–storage curve) to estimate seepage during bar-closure events.

## Getting started

### Environment

```bash
conda env create -f environment.yml
conda activate russian-river-seepage
```

### Data

* **Raw data** → `data/raw/` (API pulls or local CSV/Excel)
* **Interim** → `data/interim/` (merged/resampled tables)
* **Processed** → `data/processed/` (final tidy time-series)
* **Sample** → `data/sample/` (**only** this folder is committed)

### Configuration

Edit `configs/estuaries.yml` (station IDs, paths, dates).

### Run notebooks

1. `notebooks/10_merge_sources.ipynb` → writes `data/interim/merged_hourly.parquet`
2. `notebooks/20_seepage_mass_balance.ipynb` → writes `data/processed/estuary_timeseries.parquet`
3. `notebooks/99_figures.ipynb` → reproduces the figure below

---

## Quick figure (stage + seepage/flow)

```python
# quick_figure.py (or run in a notebook cell)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load tiny demo file committed in the repo
demo = Path(__file__).resolve().parent / "data" / "sample" / "example.parquet"
df = pd.read_parquet(demo)

# Ensure we have a datetime index
time_col = next((c for c in ["Time","time","datetime","date"] if c in df.columns), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
df = df.sort_index()

# Heuristics for common column names
stage_cols = [c for c in df.columns if c.lower() in ("h_m","stage_m","stage","water_level_m","usgs_h","visitor_h")]
seep_cols  = [c for c in df.columns if "seep" in c.lower()]
q_cols     = [c for c in df.columns if c.lower().startswith("q")]

fig, ax = plt.subplots(figsize=(10,4))

# Left axis: stage
if stage_cols:
    ax.plot(df.index, df[stage_cols[0]], label=stage_cols[0])
    ax.set_ylabel("Stage (m)")
else:
    ax.set_ylabel("Stage (m)")
    ax.text(0.02, 0.9, "No stage column found", transform=ax.transAxes, fontsize=9)

# Right axis: seepage or flow
ax2 = ax.twinx()
if seep_cols:
    ax2.plot(df.index, df[seep_cols[0]], color="C1", label=seep_cols[0])
    ax2.set_ylabel("Seepage")
elif q_cols:
    ax2.plot(df.index, df[q_cols[0]], color="C1", label=q_cols[0])
    ax2.set_ylabel("Q")
else:
    ax2.set_ylabel("Seepage / Q")
    ax2.text(0.02, 0.85, "No seepage/flow column found", transform=ax.transAxes, fontsize=9, color="C1")

ax.set_xlabel("Date")
ax.grid(True, alpha=0.25)

# Legends
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
if l1 or l2:
    ax.legend(h1+h2, l1+l2, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()
```

**No sample file yet? Create one from your processed table:**

```python
import pandas as pd
from pathlib import Path

processed = Path("data/processed/estuary_timeseries.parquet")
df = pd.read_parquet(processed)

# pick a week window with interesting behavior
cols = [c for c in ["h_m","USGS_h","visitor_h","seepage","Q_Austin","Q_Hacienda","Q"] if c in df.columns]
df_demo = df.loc["2020-01-01":"2020-01-07", cols]

Path("data/sample").mkdir(parents=True, exist_ok=True)
df_demo.to_parquet("data/sample/example.parquet", index=True)
```

---

## Repo layout

* `scripts/` – core functions (`plot_config`, `timeseries_functions`, `seepage_analysis`, `seepage_plots`)
* `notebooks/` – thin notebooks that call into `scripts/`
* `tests/` – lightweight checks for IO, merge, and mass-balance closure
* `data/` – local storage; only `sample/` is versioned

## Citation

See `CITATION.cff` once a DOI is minted.

## License

See `LICENSE`. (For USG-only contributions, include a public-domain/CC0 notice. Add partners’ terms if applicable.)
