# Singapore birth rate: what moves it? (v9)

AN6100 group project · five factors against the total fertility rate (TFR), SingStat Table Builder, 2010–2025.

## What is in this folder

| File / folder | What |
|---|---|
| `SB_Group8_Dashboard.py` | The whole Streamlit dashboard in one file. Run: `streamlit run SB_Group8_Dashboard.py` |
| `Fertility_Combined_Data.csv` | Everyone's data in one table: one row per year 2000–2025, 111 columns |
| `Dashboard_Graphs_Merged.ipynb` | All five parts in one notebook (Healthcare, Housing, Marriage, Income, Education) |
| `members/<n>_<factor>_<name>/` | Each member's own notebook and own CSV |
| `charts/` | Every chart: one folder per member (from their notebook) + `0_Dashboard/` (screenshots of each page) |
| `sections/` | Helper code and figures the combined notebook uses (Education), output folders for Healthcare / Housing |

### Members

| Folder | Notebook | Data |
|---|---|---|
| `1_Healthcare_Zhou_Dianfeng` | `Healthcare_Zhou_Dianfeng.ipynb` | `Healthcare_Data.csv` |
| `2_Housing_Angelica_Wu` | `Housing_Angelica_Wu.ipynb` | `Housing_Data.csv` |
| `3_Marriage_Tongtong_Tang` | `Marriage_Tongtong_Tang.ipynb` | `Marriage_Data.csv` |
| `4_Income_WangGaoyang` | `Income_WangGaoyang.ipynb` | `Income_WangGaoyang_Data.csv` (raw SingStat values + cleaned table), `raw/` original downloads |
| `5_Education_Huang_Yixuan` | `Education_Huang_Yixuan.ipynb` | `Education_Data.csv` |

Each member CSV holds the columns of the combined table that member's code reads (Education reads the group master
columns, as in its original `team_master.csv`). `Income_WangGaoyang_Data.csv` is long format:
`part` (raw / clean), `table` (SingStat ID), `series`, `period`, `value`, `unit`; the notebook rebuilds the clean table
from the raw rows and checks it.

## Run

```bash
pip install -r requirements.txt
streamlit run SB_Group8_Dashboard.py          # dashboard
jupyter lab                                   # notebooks: Restart Kernel and Run All
```

## Methods used everywhere

- Real income = nominal income (17872) ÷ CPI All Items (M213751, annual mean, expressed as 2010 = 100) × 100.
  This reproduces SingStat's published real change (17885) exactly in every year 2011–2024.
- Pearson r with two-sided p-value, n = years used. Levels: r of x[t] and TFR[t+L].
  Yearly change: r of x[t] − x[t−1] and TFR[t+L] − TFR[t+L−1]. Lag L = 0–3 years.
- Later-marriage shift index (Marriage) = female marriage rate 30–34 ÷ rate 25–29.

## Changes in v8

- Built on the group's `Fertility_Group_Analysis` package; Kenny's income part added as Part 4 (notebook, data, charts).
- Removed non-SingStat derived columns `Real_Income_Index`, `Nominal_Income_Index`; added real income in 2010 S$, CPI,
  17885 decile growth and M810361 top / lowest 10% growth.
- Dashboard: Income page rebuilt (nominal + real, 17885 heatmap, Gini with TFR panel, yearly Gini finding, r/p for every
  comparison); Marriage page adds "Marriage rate by age group vs TFR" and "Relative shift toward age 30–34 vs TFR";
  Overview bars and lines use the actual variable names; chart captions reduced to formula + source; every KPI is a
  value shown in a chart on the same page.

## Changes in v9

- The dashboard shows data only; insights live in the slides. Correlations appear only in each factor's last chart
  ("<Factor> vs TFR": pick a series, Levels / Yearly change, lag 0–3) and on the Correlations page.
- Removed: Income "year to year" scatter, Gini r/p table and finding, income r/p grid; Marriage "Key correlations"
  matrix, "Marriage rate by age group vs TFR" and the separate shift-index scatter; Housing "HDB prices and births by
  mother's age". The later-marriage shift index is the default series in "Marriage vs TFR" (x = 1 reference line).
