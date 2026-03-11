# Russian River Seepage

![Russian River inlet at Jenner during a bar-closure event (Jan 2, 2022). The sand berm separating the estuary from the ocean is visible in the foreground.](figures/russian_river_inlet.jpg)

Mass-balance workflow for the Russian River estuary at Jenner, CA.
This repository merges hydrologic and oceanographic records 
(USGS, NOAA CO-OPS, visitor-center gauges, inlet-state logs, and a stage–storage curve) to estimate seepage through the sand berm during bar-closure events.

## Seepage models

During inlet closures the estuary loses water to the ocean by seepage through the sand berm.
Seepage $S$ is estimated from a basin-scale water balance and modeled via Darcy's law with an effective berm conductance $C$:

$$S = C(\cdot)\,\Delta h$$

Three candidate conductance formulations are compared:

| Model | Equation | Parameters |
|---|---|---|
| Linear (constant $C$) | $S = C_L\,\Delta h$ | $C_L$ |
| Offset–power on $\Delta h$ | $S = C_\Delta\,\zeta_\Delta^{\,b_\Delta}\,\Delta h$ | $C_\Delta,\;\delta_\Delta,\;b_\Delta$ |
| Offset–power on $h_V$ | $S = C_V\,\zeta_V^{\,b_V}\,\Delta h$ | $C_V,\;\delta_V,\;b_V$ |

where $\zeta_\Delta = (\Delta h + \delta_\Delta)/m_\Delta$ and $\zeta_V = (h_V + \delta_V)/m_V$ are dimensionless scaled water levels, and $m$ denotes the median normalization.

## Getting started

### Environment

```bash
conda env create -f environment.yml
conda activate russian-river-seepage
```

### Data

* **Raw data** → `data/` (API pulls or local CSV/Excel)
* **Interim** → `data/interim/` (merged/resampled tables)
* **Processed** → `data/processed/`

## Notebooks

| Notebook | Description |
|---|---|
| `1 Russian River – merge data` | Merge and align hydrologic/oceanographic time series |
| `2 Russian River – analysis` | Exploratory analysis of merged dataset |
| `3 Seepage vs DeltaH model fits` | Three-model comparison (linear, Δh-power, $h_V$-power) with pooled and per-event fits, AIC/BIC, and bootstrap |
| `4 Crest` | Berm crest elevation analysis |
| `Model description – three models` | Derivation and notation for the three seepage models |
| `Model description – two models` | Derivation and notation for the two-model variant |

## Repository structure

```
├── data/                 # Raw and processed data
├── figures/              # Generated figures
├── figures_edited/       # Post-processed figures for publication
├── notebooks/            # Analysis notebooks
├── src/                  # Reusable Python modules
│   ├── seepage_analysis.py
│   ├── seepage_plots.py
│   ├── timeseries.py
│   └── plot_config.py
├── environment.yml       # Conda environment specification
└── README.md
```
