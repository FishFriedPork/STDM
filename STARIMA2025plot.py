# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.vectorized import contains


def idw_naive(points, values, x, y, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation for the given (x, y) coordinate.

    :param points: ndarray of shape (N, 2), observation points (longitude, latitude)
    :param values: ndarray of shape (N,), corresponding AQI values
    :param x: float, longitude of the grid point
    :param y: float, latitude of the grid point
    :param power: power parameter for distance decay, default is 2
    :return: float, interpolated AQI value at the grid point
    """
    dist = np.sqrt((x - points[:, 0]) ** 2 + (y - points[:, 1]) ** 2)
    # If any distance is zero, return the corresponding observed value directly
    if np.any(dist == 0):
        return values[dist == 0][0]
    # Compute weights
    w = 1 / (dist ** power)
    return np.sum(w * values) / np.sum(w)


def main():
    # ------------------------------------------------------
    # 1. Read AQI data for June 1, 2025, from STARIMA_Predict.xlsx
    # ------------------------------------------------------
    aqi_file = r"C:\Users\于嘉诚\Desktop\STDM\STARIMA_Predict.xlsx"

    # Read the Excel file. If there's a problematic last line (e.g. "Evaluation Metrics"),
    # you can add skipfooter=1 to skip it. Adjust if needed:
    # df_all = pd.read_excel(aqi_file, skipfooter=1, engine='openpyxl')
    df_all = pd.read_excel(aqi_file, engine='openpyxl')

    # Convert the "Date" column to datetime and filter out invalid dates
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
    df_all = df_all.dropna(subset=['Date'])

    # Filter data for 2025-06-01
    target_date = pd.Timestamp("2025-06-01")
    df_target = df_all[df_all['Date'].dt.normalize() == target_date]

    if df_target.empty:
        print("No data found for 2025-06-01. Please check the 'Date' column in the Excel file.")
        return

    # Convert the AQI row to a (station -> AQI) mapping
    # Assuming each column (besides 'Date') represents a station's AQI
    aqi_series = df_target.drop(columns=["Date"]).iloc[0]
    df_aqi = pd.DataFrame({
        'station': aqi_series.index,
        'AQI': aqi_series.values
    })

    # ------------------------------------------------------
    # 2. Read station geographic information
    # ------------------------------------------------------
    station_file = r"C:\Users\于嘉诚\Desktop\STDM\raw data\北京市站点列表-2021.01.23起.xlsx"
    df_station = pd.read_excel(station_file)
    df_station.rename(columns={
        "监测点": "station",
        "经度": "lon",
        "纬度": "lat"
    }, inplace=True)

    # Merge station data with AQI data
    df_merged = pd.merge(df_station, df_aqi, on="station", how="inner")
    if df_merged.empty:
        print("Merged data is empty. Please check if station names match.")
        return

    # ------------------------------------------------------
    # 3. Read the Beijing boundary shapefile
    # ------------------------------------------------------
    shapefile_path = r"C:\Users\于嘉诚\Desktop\STDM\Beijing location map\Beijing.shp"
    beijing = gpd.read_file(shapefile_path)
    lon_min, lat_min, lon_max, lat_max = beijing.total_bounds

    # ------------------------------------------------------
    # 4. Perform IDW interpolation
    # ------------------------------------------------------
    points = df_merged[["lon", "lat"]].values
    values = df_merged["AQI"].values

    grid_resolution = 500
    grid_x = np.linspace(lon_min, lon_max, grid_resolution)
    grid_y = np.linspace(lat_min, lat_max, grid_resolution)
    mesh_lon, mesh_lat = np.meshgrid(grid_x, grid_y)

    grid_idw = np.empty_like(mesh_lon, dtype=float)
    M, N = mesh_lon.shape
    for i in range(M):
        for j in range(N):
            x = mesh_lon[i, j]
            y = mesh_lat[i, j]
            grid_idw[i, j] = idw_naive(points, values, x, y, power=2)

    # ------------------------------------------------------
    # 5. Mask the result to keep only data within Beijing
    # ------------------------------------------------------
    # Use the newer union_all() instead of unary_union
    mask = contains(beijing.union_all(), mesh_lon, mesh_lat)
    grid_idw_masked = np.where(mask, grid_idw, np.nan)

    # ------------------------------------------------------
    # 6. Plot and save the Beijing AQI map
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    beijing.boundary.plot(ax=ax, color='black', linewidth=1)
    contour = ax.contourf(mesh_lon, mesh_lat, grid_idw_masked, levels=50,
                          cmap='RdYlGn_r', alpha=0.8)
    cbar = fig.colorbar(contour, ax=ax, label="AQI (IDW Interpolated)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save the image to the specified folder, named STARIMA2025plot.png
    output_path = r"C:\Users\于嘉诚\Desktop\STDM\STARIMA2025plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
