"""Central configuration for the ML pipeline: paths, class names and tunable thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = ROOT / "models"
METRICS_DIR = MODELS_DIR / "metrics"
DOCS_DIR = ROOT / "docs"

SEGMENTATION_WEIGHTS = MODELS_DIR / "segmentation_best.pth"
# Deep ensemble: average the predictions of these trained models (newest file per pattern).
# Where they disagree, the detection is uncertain. Set USE_ENSEMBLE = False to use only the best model.
USE_ENSEMBLE = True
ENSEMBLE_MODELS = ["sos_unet_*.pth", "sos_deeplabv3plus_*.pth", "sos_segformer_*.pth"]
DISAGREEMENT_FLAG = 0.15       # mean spread of oil probability between models that counts as "models disagree"
RANKER_PATH = MODELS_DIR / "ranker.pkl"
COASTLINE_PATH = REFERENCE_DIR / "coastline.geojson"
LAND_PATH = REFERENCE_DIR / "land.geojson"
SENSITIVE_SITES_PATH = REFERENCE_DIR / "sensitive_sites.csv"

# Krestenitis et al. (2019) dataset classes (labels_1D values).
# Check the dataset README after download; change here if the order differs.
CLASSES = ["sea", "oil_spill", "look_alike", "ship", "land"]
OIL_CLASS = 1
LOOKALIKE_CLASS = 2

# ---------------- Detection ----------------
MAX_SCENE_SIZE = 2048          # longest side (pixels) after downsampling a SAR scene
MIN_SPILL_PIXELS = 60          # ignore tiny blobs
TILE_SIZE = 512
TILE_OVERLAP = 64

# Detection reliability checks (radar oil detection works best at roughly 3-10 m/s wind;
# in calm water the whole sea looks dark and is easily mistaken for oil)
MIN_RELIABLE_WIND_MS = 3.0
MAX_RELIABLE_WIND_MS = 10.0   # above this, slicks break up and rain cells / wind shadows cause most dark patches
MAX_SPILL_KM2 = 150.0          # a dark area larger than this in calm OR very strong wind is treated as a look-alike

# ---------------- AIS / attribution ----------------
ATTRIBUTION_WINDOW_H = 12      # look this many hours before the image
SEARCH_RADIUS_KM = 25          # candidate vessels must pass this close to the (back-tracked) spill
OVERLAP_KM = 1.0               # track counts as overlapping spill axis within this distance
AIS_GAP_FLAG_H = 0.5           # gap longer than this near the spill is suspicious
LOITER_SPEED_KN = 3.0

# Responders are near a spill BECAUSE of it - never list them as suspects.
# AIS ship types: 35 military, 51 search & rescue, 53 port tender, 54 anti-pollution, 55 law enforcement, 58 medical
RESPONSE_TYPE_CODES = {35, 51, 53, 54, 55, 58}
RESPONSE_NAME_KEYWORDS = ["FIREBOAT", "FIRE BOAT", "RESCUE", "COAST GUARD", "USCG", "POLICE", "PATROL",
                          "PILOT", "RESPONDER", "OCEAN DEFENDER", "TOWBOATUS", "SEA TOW"]

RULE_WEIGHTS = {
    "proximity": 0.30,
    "alignment": 0.20,
    "overlap": 0.20,
    "recency": 0.10,
    "type_risk": 0.10,
    "behaviour": 0.10,
}

# ---------------- Drift / prediction ----------------
WINDAGE = 0.03                 # oil moves at ~3% of wind speed
DIFFUSION_M2S = 10.0           # horizontal random-walk diffusion
N_PARTICLES = 400
FORECAST_HOURS = [6, 12, 24, 48, 72]
DRIFT_STEP_MIN = 30
BEACH_DISTANCE_KM = 0.5        # particle within this distance of coast = beached

# Default environment if no wind/current files are available (m/s, eastward u / northward v)
DEFAULT_ENV = {"wind_u": 3.0, "wind_v": 1.0, "current_u": 0.05, "current_v": 0.05}

# ---------------- Impact ----------------
# Bonn Agreement appearance codes span ~0.04 to >200 micrometres.
# SAR cannot tell thickness, so we report a low/high range.
THICKNESS_UM_LOW = 0.3
THICKNESS_UM_HIGH = 50.0
SITE_THREAT_KM = 3.0
