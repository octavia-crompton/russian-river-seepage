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
* **Processed** → `data/processed/` 
