"""Download wind (ERA5) and surface currents (Copernicus Marine) for the drift / prediction module.

One-time setup:
  * ERA5: free account at https://cds.climate.copernicus.eu , accept the ERA5 licence, then create
    the file  C:\\Users\\<you>\\.cdsapirc  with two lines:
        url: https://cds.climate.copernicus.eu/api
        key: <your personal access token>
    pip install cdsapi
  * Currents: free account at https://data.marine.copernicus.eu , then
    pip install copernicusmarine   and run   copernicusmarine login   once.

    python scripts/download_weather.py --start 2021-10-01 --end 2021-10-06
Files are saved as Datasets/raw/weather/era5_wind.nc and Datasets/raw/weather/cmems_current.nc
(the pipeline picks them up automatically). Without them the model uses constant values.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

OUT = Path("Datasets/raw/weather")
AREA = (-118.8, 33.0, -117.4, 34.1)   # min_lon, min_lat, max_lon, max_lat (Huntington Beach region)


def days_between(start, end):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def era5(start, end, area):
    import cdsapi
    days = days_between(start, end)
    months = sorted({d.strftime("%m") for d in days})
    years = sorted({d.strftime("%Y") for d in days})
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": years, "month": months, "day": sorted({d.strftime("%d") for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [area[3], area[0], area[1], area[2]],     # north, west, south, east
        "data_format": "netcdf",
    }, str(OUT / "era5_wind.nc"))
    print("Saved Datasets/raw/weather/era5_wind.nc")


def currents(start, end, area):
    import copernicusmarine
    # GLORYS12 multi-year reanalysis (daily). If the id changes, search "GLOBAL_MULTIYEAR_PHY_001_030"
    # on data.marine.copernicus.eu and paste the new dataset id here.
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=area[0], maximum_longitude=area[2],
        minimum_latitude=area[1], maximum_latitude=area[3],
        start_datetime=f"{start}T00:00:00", end_datetime=f"{end}T23:59:59",
        minimum_depth=0.0, maximum_depth=1.0,
        output_directory=str(OUT), output_filename="cmems_current.nc",
    )
    print("Saved Datasets/raw/weather/cmems_current.nc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default="2021-10-06")
    ap.add_argument("--only", choices=["wind", "current"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "wind"):
        era5(args.start, args.end, AREA)
    if args.only in (None, "current"):
        currents(args.start, args.end, AREA)
