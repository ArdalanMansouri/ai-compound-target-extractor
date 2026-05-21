# AI Compound Target Extractor

A Python toolkit for automated extraction of molecular targets from compound screening data using the Perplexity AI API, with built-in visualization and analysis utilities.

## Background 

After performing high-throughput screening, in order to find the mechanism of
action (MoA) of the compounds, we need to know the molecular targets of the compounds. 
Nevertheless, searching through the literature for each compound is time-consuming and inefficient, and we can rely on a scientific AI to mine the literature and extract
the relevant information for us. Nevertheless, the input token for the API of most 
AI models is limited, and we cannot query the API for all the compounds at once. 
Therefore in this project, we will automate the process of querying the API for a list of compounds (as an example), and we will extract the relevant information for each compound and organize it in an Excel file. This then is used for generating graphs and tables for the analysis of MoA of the compounds.

## Overview

This project supports understanding the effect of compounds from high-throughput compound libraries. The steps are:

1. **Query the Perplexity AI API**: Uses the SDK of Perplexity to extract known molecular targets and biological keywords for each compound, and export the 
results to an Excel file.

2. **Evaluate the frequency of words related to mechanism of action**: In order to 
group the compounds together, we need to find a common grouping factor (keyword) for the compounds. This can be done either manually by the judgment of the researcher or automatically using the function `word_frequency`, which will evaluate the frequency of keywords related to MoA across the selected column of a dataframe (e.g., column with
explanations of MoA).

3. **Annotates the compound dataset**: Add the keywords to the relevant 
compounds.

4. **Visualizes keyword distributions**: Generate interactive plotly graph to show
the distribution of keywords across compound types (Normal, Inhibitor, Inducer).

5. **Ranks and tables compounds**: Generate an elaborated graph with details about 
the rank of the compounds based on their strength of effect on the uptake of
extracellular vesicles (EVs). A table is generated with the same order of the 
inhibitor and inducer compounds in the graph, so that the user can easily find the compounds of interest and their ranking. 

6. **Highlight the compounds of interest**: The user can provide a list of compounds of interest, and the function `compound_finder` will search for these compounds in the ranked table and mark them with an [X] in the table, and print the number of hits found.


## Project Structure

```
ai-compound-target-extractor/
├── pyproject.toml                        # Package metadata and dependencies
├── .env                                  # API key (not tracked by git)
├── data/
│   ├── compounds_names.xlsx              # Input list of compound names
│   ├── tested_outliers.xlsx              # The compounds that were validated in the lab 
├── src/
    └── target_extractor/
        ├── __init__.py
        ├── perplex_sdk_pro.py            # Perplexity API via official SDK 
        ├── graph_gen.py                  # Graph class: visualization and table methods
├── notebooks/
│   ├── target_extractor_with_prompt.ipynb  # Query Perplexity API and export results
│   └── prevalent_targets_graph.ipynb       # Main analysis and visualization notebook
└── results/
    ├── masterfile.xlsx                   # Outcome of target_extractor_with_prompt.ipynb
    ├── target_prevalence_graph.html      # Interactive prevalence graph
    └── bar_graph_table.xlsx              # Ranked compound table export

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

### `perplex_sdk_pro.py`

A module for querying the Perplexity AI API (`sonar-pro` model, academic search mode) with exponential backoff. The main function `query_perplexity(compounds)` takes a list of compound names, sends them in batches to the API, and returns a DataFrame with extracted targets and keywords.

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
