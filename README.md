# Marine Trace AI (SpillWatch): Oil Spill Detection and Vessel Attribution

MCA final-year project (Machine Learning), Centurion University of Technology and Management.
Problem statement: Smart India Hackathon **SIH26143**: detect oil spills in satellite radar images and identify the ship most likely responsible, using AIS ship-tracking data.

**Author:** Soumyaranjan Satapathy · [GitHub](https://github.com/Soumya1225) · [LinkedIn](https://linkedin.com/in/soumyaranjan-satapathy/)

---

## What it does

1. **Checks the input image**: a picture that is not a satellite radar image of the sea (for example a phone photo) is rejected before any model runs; a radar image gets an instant verdict: oil, no oil, or possibly a look-alike.
2. **Detects oil spills** in Sentinel-1 SAR images with a deep ensemble of three segmentation models (U-Net, DeepLabV3+, SegFormer). Where the models disagree, the spill is flagged as uncertain.
3. **Checks reliability**: masks land, and warns when the wind is too low (calm water looks like oil in radar).
4. **Second-stage look-alike check**: every detected patch is scored oil vs look-alike from physical evidence the deep models do not see directly (wind, contrast, edge sharpness, shape, texture, model agreement), with a plain-language reason for each piece of evidence.
5. **Finds ships without AIS**: bright radar targets with no matching AIS position (possible "dark" vessels). Targets that sit at the same spot in two images hours apart, or on a known platform position, are marked as fixed structures (oil platforms).
6. **Platform and pipeline layer (GIS)**: offshore platforms and pipelines are shown on the map; when a spill or its back-tracked drift path comes close to one, the incident is flagged as a **possible non-ship source** and an "Inspect platform / pipeline" action is added.
7. **Back-tracks the spill** with a Lagrangian drift model driven by real ERA5 wind and Copernicus Marine currents.
8. **Ranks suspect ships** from AIS tracks (distance, timing, course alignment, AIS gaps, slowing down, loitering, vessel type) with a rule score blended with an XGBoost ranker. **SHAP** explains, for every ship, which evidence raised or lowered its score. Rescue, fire, pilot and anti-pollution vessels are excluded automatically.
9. **Forecasts drift** for 6-72 h (physics ensemble with an uncertainty zone, plus an LSTM forecast) and estimates **landfall and threatened sensitive sites**.
10. **Decision support**: priority level, recommended actions, alerts, and a PDF report with SHA-256 evidence fingerprints of the input files.
11. **Hotspots and trends**: incidents are grouped into 10 km grid cells to show pollution hotspots, with a trend chart split by the look-alike verdict.
12. **Feedback loop**: incidents confirmed or rejected and suspects confirmed or dismissed by analysts are exported as new training data (`Datasets/feedback/`), and `scripts/retrain_from_feedback.py` measures how well the system agrees with analysts and learns look-alike weights from their decisions.

Everything is available through a web dashboard with role-based login (admin, analyst, viewer).

## Results

### Spill detection (SOS Sentinel-1 dataset, test set, 839 images)

| Model | Oil IoU | Oil F1 |
|---|---|---|
| FCN baseline | 0.668 | 0.801 |
| U-Net + ResNet34 | **0.770** | **0.870** |
| SegFormer MiT-B1 (transformer) | 0.769 | 0.869 |
| Swin-Tiny (transformer) | 0.767 | 0.868 |
| DeepLabV3+ + ResNet50 | 0.765 | 0.867 |
| **Deployed:** deep ensemble of U-Net, DeepLabV3+ and SegFormer | | |

The four strong models are within 0.5% of each other: on this dataset the limit is the data, not the architecture.

**Input check:** 100 of 100 real radar images pass the radar-image gate; on labelled crops, all 96 oil images were found.

### Vessel attribution (1,000 synthetic cases with a known source, 250 held out for testing)

| Ranker | Top-1 | Top-3 |
|---|---|---|
| Rule-based score | 57% | 69% |
| **XGBoost** (deployed) | **72%** | **88%** |
| LightGBM LambdaRank (learning-to-rank) | 70% | 86% |
| Cross-attention model | 70% | 83% |

With 300 training cases the cross-attention model reached only 55% Top-1; with 1,000 cases it reached 70%, so the attention model needs more data, while XGBoost is accurate and stable with little data.

### LSTM drift model (120 held-out trajectories, real ERA5 / Copernicus forcing)

| Horizon | LSTM error | Persistence baseline |
|---|---|---|
| 6 h | 0.06 km | 1.18 km |
| 24 h | 0.07 km | 6.13 km |

The training trajectories come from the physics drift model, so the LSTM is a fast learned version of that model, not a replacement validated against real oil.

### Case study: Huntington Beach oil spill, California, 3 October 2021

Two real Sentinel-1 images with real NOAA AIS data:

- **Morning image (01:49 UTC, wind 4.7 m/s):** 5 spills detected; the main slick (about 4.4 km²) was marked Critical, Newport Beach flagged as already within reach of the oil, and responder vessels correctly excluded from the suspects.
- **Afternoon image (13:51 UTC, wind 0.6 m/s):** marked **low reliability** because of calm sea; the system recommends verification.
- **Forecast check:** the afternoon image shows no slick within 10 km of any morning forecast, explained by the calm sea (oil cannot be separated from flat water in radar), so this image pair cannot validate the forecast.
- **82 bright targets** in the afternoon radar image, 62 matched to AIS ships (76%); 3 sit on known oil platform positions and are no longer counted as dark vessels.
- **Look-alike check:** the main morning slick scores **Likely oil (0.70)**, while the calm-sea afternoon patch scores **Likely look-alike**.
- The real spill came from a **damaged pipeline (P00547)**, not a ship. The system spreads suspicion across many vessels instead of blaming one ship with high confidence, and the platform and pipeline layer lets the investigator check the installations as well.

## Quick start (Windows)

Requirements: Python 3.11+ and Node.js 18+.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt -r backend/requirements.txt
cd frontend && npm install && cd ..
```

Then double-click **start.bat**: it starts the backend (http://127.0.0.1:8000) and the dashboard (http://localhost:5173) and opens the browser.

Manual start: `cd backend && uvicorn app.main:app --reload` and, in a second terminal, `cd frontend && npm run dev`.

The first start creates an admin account (`admin@oilspill.local` / `admin123`). Change the password on the Users page.

Use **New analysis → Create demo data** to try the system with synthetic data.

## Project structure

```
Datasets/         all data in one place (see Datasets/README.md): satellite, labels, AIS, environmental, GIS, synthetic, feedback
ml/
  segmentation/   dataset, models (FCN, U-Net, DeepLabV3+, SegFormer, Swin-Tiny), training, ensemble prediction
  lookalike/      second-stage oil vs look-alike check
  preprocessing/  Sentinel-1 loading (GCP warp to north-up), land mask, AIS cleaning
  features/       spill shape, AIS track features, ship detection, fixed structures, platforms and pipelines
  attribution/    candidate filter, drift model, scorer + SHAP, XGBoost / LightGBM / cross-attention rankers, LSTM drift
  impact/         landfall and threatened-site assessment
  decision/       priority, actions, alerts
  analytics/      pollution hotspots and trends
  validation/     forecast check between two images of the same area
  gis/            offshore platform and pipeline layer (GeoJSON)
  quickcheck.py   radar-image gate and instant verdict
  pipeline.py     end-to-end analysis
backend/app/      FastAPI + SQLite API, background jobs, PDF reports, feedback export
frontend/         React + Vite + Leaflet dashboard
models/           trained models (weights not in GitHub) and metrics/
scripts/          data download, maintenance (reset, mark platforms, look-alike check), feedback retraining
notebooks/        Google Colab training notebook
docs/             screenshots and design decisions
reports/          figures and generated reports
tests/            pytest suite
start.bat         one-click start
```

## Data

All data is kept in `Datasets/` (see `Datasets/README.md` for every dataset, its source and where it is used). Large files are not stored in this repository.

| Data | Source |
|---|---|
| Sentinel-1 GRD images | NASA ASF / Copernicus |
| AIS ship tracks | NOAA MarineCadastre |
| Wind (ERA5) and ocean currents | Copernicus CDS / Copernicus Marine |
| Coastline and land | Natural Earth |
| Training images | SOS Deep-SAR Oil Spill dataset |
| Platforms and pipelines | NOAA IncidentNews, BOEM Pacific OCS maps (approximate positions) |

Model weights (`models/*.pth`) are too large for GitHub. Segmentation models were trained on Google Colab (T4 GPU) with `notebooks/train_on_colab.ipynb`. Attribution and LSTM models train on a laptop:

```bash
python -m ml.attribution.lstm_drift
python -m ml.attribution.train_ranker
```

## Tests

```bash
python -m pytest tests -q
```

## Limitations

- Attribution models are trained on synthetic cases, because real spill cases with a confirmed responsible ship and matching AIS data are rare.
- SAR pixel values use an approximate dB scale (no full radiometric calibration); each image is normalised before detection.
- The training data has only oil / sea classes, so look-alikes are handled by a second-stage check whose thresholds come from the SAR literature; the feedback loop can learn better weights once enough incidents are reviewed.
- Platform and pipeline positions are approximate (digitised from public maps); replace `ml/gis/offshore_infrastructure.geojson` with official BOEM layers for operational use.
- Radar cannot measure oil thickness, so the volume estimate is a wide range.
- Ships without AIS can include small boats that are not required to carry AIS.

## Data credits

Contains modified Copernicus Sentinel data 2021. AIS data: NOAA Office for Coastal Management / MarineCadastre.gov. Wind: ERA5 (Copernicus Climate Change Service). Currents: Copernicus Marine Service. Maps: © OpenStreetMap contributors.
