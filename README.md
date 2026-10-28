# From Pilots to Agents: How US Banks Talk About AI (2019–2026)

DATS 6401 · Mid-Semester Project · Multi-Modality Visualization

An interactive Streamlit app about how US commercial banks write about **AI, Generative AI and Agentic AI**
in their annual reports (SEC form 10-K), 2019–2026.

## Four data modalities
| Tab | Modality | Main view |
|---|---|---|
| Over time | Time series | % of reports mentioning AI per year, 95% intervals |
| What banks say | Text | word cloud, top words, themes, keyness |
| Network | Network | banks linked by similar AI language, Louvain groups |
| Map | Map | classed choropleth (equal-area projection) |

## Data (free, public)
- **SEC EDGAR** full-text search and 10-K documents; total assets from the SEC XBRL *frames* API.
- **FDIC BankFind** — map position of bank offices.

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the data
Run `ai_banks_project.ipynb` from top to bottom (first run downloads ~2,600 reports and takes 4–5 hours;
later runs use the cache in `raw/`). It writes every file in this folder.

## Accessibility
Colour-blind-safe colours (Okabe–Ito, ColorBrewer YlGnBu) · chart titles state the finding · axis labels with units ·
alt text for every chart · no information shown by colour alone (labels and hover text).

