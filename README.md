# Thailand Hotel Recommender

Content-based hotel recommender for Thailand (step-by-step Q&A) built on 977 hotels.
Pipeline: data cleaning -> K-Means (k=5) -> decision-tree explanation -> composite score -> cosine-similarity recommendation -> offline evaluation.

## Run the web app
```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit front-end |
| `recommender.py` | features, cosine similarity, ranking, explanations |
| `thailand_hotels_clustered.csv` | cleaned data + cluster + composite scores |
| `evaluate.py`, `make_figures.py` | evaluation tables and slide figures (`pip install -r requirements-analysis.txt`) |
| `train_test_demo.py`, `train_test_weights.py`, `weight_sweep.py` | 80/20 train-test demo, weight selection |
| `ahp_weights.py`, `human_eval_sheet.csv` | weight derivation with AHP, human evaluation form |

## Limitations
977 hotels, one sample room per hotel; prices are estimates; room type is grouped from room-name keywords;
only 10 pool villas; scoring weights are hand-set defaults; evaluated with simulated queries.
