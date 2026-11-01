"""Select vessels that passed close to the (back-tracked) spill before the image was taken."""
import numpy as np
import pandas as pd

from ml.config import ATTRIBUTION_WINDOW_H, SEARCH_RADIUS_KM
from ml.utils.geo import haversine_km


def backtrack_offsets(back_hours, back_lat, back_lon):
    """Return function hours_before -> (dlat, dlon) shift of the spill at that time."""
    dlat = back_lat - back_lat[0]
    dlon = back_lon - back_lon[0]

    def offset(hours_before):
        h = np.clip(hours_before, back_hours[0], back_hours[-1])
        return np.interp(h, back_hours, dlat), np.interp(h, back_hours, dlon)

    return offset


def candidate_vessels(ais, spill, image_time, offset_fn, window_h=ATTRIBUTION_WINDOW_H,
                      radius_km=SEARCH_RADIUS_KM):
    """Return AIS rows (with 'hours_before' column) of vessels that came within radius_km."""
    image_time = pd.Timestamp(image_time)
    df = ais[(ais.time >= image_time - pd.Timedelta(hours=window_h)) &
             (ais.time <= image_time + pd.Timedelta(minutes=30))].copy()
    if df.empty:
        return df
    df["hours_before"] = (image_time - df.time).dt.total_seconds() / 3600.0
    dlat, dlon = offset_fn(df.hours_before.clip(lower=0).values)
    df["dist_centroid_km"] = haversine_km(df.lat.values, df.lon.values,
                                          spill["centroid_lat"] + dlat, spill["centroid_lon"] + dlon)
    near = df.groupby("mmsi").dist_centroid_km.min()
    keep = near[near <= radius_km + spill.get("length_km", 0) / 2].index
    return df[df.mmsi.isin(keep)]
