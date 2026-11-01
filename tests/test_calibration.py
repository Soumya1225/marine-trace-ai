"""Tests for Sentinel-1 radiometric calibration (B8): calibration / noise XML parsing and sigma0.
No rasterio or real image needed:
    python -m pytest tests/test_calibration.py -q
"""
import numpy as np
import pytest

from ml.preprocessing.sar_preprocess import (_annotation_names, azimuth_noise_grid, calibrate_sigma0, lut_grid,
                                             parse_calibration, parse_noise)

CAL_XML = """<?xml version="1.0"?><calibration><calibrationVectorList count="2">
<calibrationVector><azimuthTime>t</azimuthTime><line>0</line><pixel count="3">0 100 200</pixel>
<sigmaNought count="3">500 600 700</sigmaNought><betaNought count="3">1 1 1</betaNought></calibrationVector>
<calibrationVector><azimuthTime>t</azimuthTime><line>100</line><pixel count="3">0 100 200</pixel>
<sigmaNought count="3">700 800 900</sigmaNought><betaNought count="3">1 1 1</betaNought></calibrationVector>
</calibrationVectorList></calibration>"""

NOISE_NEW = """<?xml version="1.0"?><noise>
<noiseRangeVectorList count="1"><noiseRangeVector><line>0</line><pixel count="2">0 200</pixel>
<noiseRangeLut count="2">100 300</noiseRangeLut></noiseRangeVector></noiseRangeVectorList>
<noiseAzimuthVectorList count="1"><noiseAzimuthVector><swath>IW1</swath><firstAzimuthLine>0</firstAzimuthLine>
<firstRangeSample>0</firstRangeSample><lastAzimuthLine>100</lastAzimuthLine><lastRangeSample>99</lastRangeSample>
<line count="2">0 100</line><noiseAzimuthLut count="2">1.0 2.0</noiseAzimuthLut></noiseAzimuthVector></noiseAzimuthVectorList>
</noise>"""

NOISE_OLD = """<?xml version="1.0"?><noise><noiseVectorList count="1"><noiseVector><line>0</line>
<pixel count="2">0 200</pixel><noiseLut count="2">50 50</noiseLut></noiseVector></noiseVectorList></noise>"""


def test_calibration_lut_is_interpolated_in_both_directions():
    vec = parse_calibration(CAL_XML)
    assert len(vec) == 2 and vec[1][0] == 100
    g = lut_grid(vec, np.array([0.0, 50.0, 100.0]), np.array([0.0, 100.0, 150.0]))
    assert g[0].tolist() == [500, 600, 650]
    assert g[1].tolist() == [600, 700, 750]          # halfway between the two lines
    assert g[2].tolist() == [700, 800, 850]


def test_sigma0_formula_and_noise_floor():
    vec = parse_calibration(CAL_XML)
    dn = np.array([[600.0, 0.0]])
    s0 = calibrate_sigma0(dn, np.array([0.0]), np.array([100.0, 100.0]), vec)
    assert s0[0, 0] == pytest.approx(600.0 ** 2 / 600.0 ** 2)   # DN^2 / A^2 = 1 -> 0 dB
    assert s0[0, 1] == 0                                         # no-data stays 0
    noisy = calibrate_sigma0(np.array([[1.0]]), np.array([0.0]), np.array([100.0]), vec,
                             parse_noise(NOISE_OLD))
    assert noisy[0, 0] == pytest.approx(1e-5)                    # DN^2 below noise -> floor (-50 dB)


def test_noise_tables_new_and_old_layouts():
    rng, az = parse_noise(NOISE_NEW)
    assert len(rng) == 1 and len(az) == 1 and az[0]["last_col"] == 99
    f = azimuth_noise_grid(az, np.array([0.0, 50.0, 100.0]), np.array([10.0, 150.0]))
    assert f[:, 0].tolist() == [1.0, 1.5, 2.0] and f[:, 1].tolist() == [1.0, 1.0, 1.0]
    rng_old, az_old = parse_noise(NOISE_OLD)
    assert len(rng_old) == 1 and az_old == []


def test_annotation_files_matched_to_the_vv_measurement():
    names = ["S1A.SAFE/measurement/s1a-iw-grd-vv-20211003t015016-001.tiff",
             "S1A.SAFE/annotation/calibration/calibration-s1a-iw-grd-vh-20211003t015016-002.xml",
             "S1A.SAFE/annotation/calibration/calibration-s1a-iw-grd-vv-20211003t015016-001.xml",
             "S1A.SAFE/annotation/calibration/noise-s1a-iw-grd-vv-20211003t015016-001.xml"]
    cal, noise = _annotation_names(names, names[0])
    assert cal.endswith("calibration-s1a-iw-grd-vv-20211003t015016-001.xml")
    assert noise.endswith("noise-s1a-iw-grd-vv-20211003t015016-001.xml")
    cog, _ = _annotation_names(names, "S1A_COG.SAFE/measurement/s1a-iw-grd-vv-20211003t015016-001-cog.tiff")
    assert cog.endswith("calibration-s1a-iw-grd-vv-20211003t015016-001.xml")
