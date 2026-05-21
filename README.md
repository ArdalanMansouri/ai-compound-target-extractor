# AI Compound Target Extractor

A Python toolkit for automated extraction of molecular targets from compound screening data using the Perplexity AI API, with built-in visualization and analysis utilities.

## Background 

## Overview

This project supports high-throughput compound library screening workflows. Given a master spreadsheet of screened compounds (e.g. Prestwick, LOPAC libraries), it:

1. **Queries the Perplexity AI API** (via SDK or HTTP requests) to extract known molecular targets and biological keywords for each compound.
2. **Annotates the compound dataset** with the returned target/keyword information.
3. **Visualizes keyword distributions** across compound types (Normal, Inhibitor, Inducer) using interactive Plotly charts.
4. **Ranks and tables compounds** by their effect on EV (extracellular vesicle) uptake, scored as log₂ fold-change relative to library controls.
5. **Searches compound lists** against the ranked table, annotating matches for downstream prioritization.

## Project Structure

```
ai-compound-target-extractor/
├── pyproject.toml                   # Package metadata and dependencies
├── data/
│   └── guide_ardalan.txt            # Notes on the master data file
├── notebooks/
│   └── prevalent_targets_graph.ipynb  # Main analysis notebook
├── perplexity_SDK_Ardalan.ipynb     # Notebook for Perplexity SDK exploration
└── src/
    └── target_extractor/
        ├── __init__.py
        ├── graph_gen.py             # Graph class: visualization and table methods
        ├── perplex_api_key.py       # API key loading utility
        ├── perplex_requests_pro.py  # Perplexity API via HTTP requests
        └── perplex_sdk_pro.py       # Perplexity API via official SDK (recommended)
```

## Key Components

### `graph_gen.py` — `Graph` class

The central analysis class. Instantiate with a compound DataFrame:

```python
from target_extractor.graph_gen import Graph

g = Graph(df_master, type_col="Screen: Effect on EV uptake")
```

| Method | Description |
|---|---|
| `bar_plot()` | Two-panel keyword frequency chart (count + stacked % by type) |
| `scored_bar_plot()` | Stacked bar chart with perceptual color gradient encoding log₂FC strength |
| `scored_table()` | DataFrame of compounds ranked strongest inducer → strongest inhibitor |
| `compound_finder(compounds)` | Substring-matches a compound list against the ranked table; marks hits with `[X]` and prints match count |

### `perplex_sdk_pro.py` / `perplex_requests_pro.py`

Two interchangeable interfaces for querying the Perplexity AI API (`sonar-pro` model, academic search mode) with exponential backoff. The SDK version (`perplex_sdk_pro.py`) supports streaming and is recommended.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
```

Dependencies (installed automatically):

- `pandas >= 2.0.0`
- `plotly >= 5.0.0`
- `requests >= 2.28.0`
- `perplexityai >= 0.1.0`

## Usage Example

```python
import pandas as pd
from target_extractor.graph_gen import Graph

df_master = pd.read_excel("data/masterfile.xlsx")
df_master["Screen: Effect on EV uptake"] = df_master["Screen: Effect on EV uptake"].fillna("Normal")

g = Graph(df_master, type_col="Screen: Effect on EV uptake")

# Interactive scored bar chart
fig = g.scored_bar_plot()
fig.show()

# Ranked compound table
df_table = g.scored_table()

# Search for compounds of interest
df_hits = g.compound_finder(["Imatinib", "Rapamycin", "Verapamil"])
```

## Data

The primary input is `Masterfile_all_compounds_updated_with_new_targets.xlsx` — a compound screening master file containing EV-uptake measurements, compound types, library assignments, and AI-extracted keyword/target annotations. This file is not included in the repository.

## License

See [LICENSE](LICENSE).
