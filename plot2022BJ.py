# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.vectorized import contains


def idw_naive(points, values, x, y, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation for the given (x, y) coordinate.

    :param points: ndarray of shape (N, 2), observation points (lon, lat)
    :param values: ndarray of shape (N,), observation values (AQI)
    :param x: float, longitude of the grid point
    :param y: float, latitude of the grid point
    :param power: distance decay power, default is 2

    :return: float, interpolated value at the grid point
    """
    # Calculate the distance from (x, y) to all observation points
    dist = np.sqrt((x - points[:, 0]) ** 2 + (y - points[:, 1]) ** 2)

    # If any distance is zero (i.e., the grid point coincides with an observation), return that observation's value
    if np.any(dist == 0):
        return values[dist == 0][0]

    # Calculate weights: w_i = 1 / (d^power)
    w = 1 / (dist ** power)

    # Compute the weighted average
    return np.sum(w * values) / np.sum(w)


def main():
    # -----------------------------
    # 1. Read the daily average AQI data for the specified date
    # -----------------------------
    aqi_file = r"C:\Users\于嘉诚\Desktop\STDM\beijing_daily_avg_aqi_all_years.csv"
    df_all = pd.read_csv(aqi_file, encoding='utf-8')

    # Convert the Date column to string and filter for data on 2022/1/1
    df_all['Date'] = df_all['Date'].astype(str)
    df_20220101 = df_all[df_all['Date'] == "20220101"]
    if df_20220101.empty:
        print("Data for 2022/1/1 not found. Please check the Date column in the CSV file.")
        return

    # Organize the AQI data into a form: station -> AQI
    aqi_series = df_20220101.drop(columns=["Date"]).iloc[0]
    df_aqi = pd.DataFrame({
        'station': aqi_series.index,
        'AQI': aqi_series.values
    })

    # -----------------------------
    # 2. Read station latitude and longitude information
    # -----------------------------
    station_file = r"C:\Users\于嘉诚\Desktop\STDM\raw data\北京市站点列表-2021.01.23起.xlsx"
    df_station = pd.read_excel(station_file)
    df_station.rename(columns={
        "监测点": "station",
        "经度": "lon",
        "纬度": "lat"
    }, inplace=True)

    # Merge the data to obtain (lon, lat, AQI)
    df_merged = pd.merge(df_station, df_aqi, on="station", how="inner")
    if df_merged.empty:
        print("No data after merging. Please check if station names match.")
        return

    # -----------------------------
    # 3. Read the Beijing region shapefile and its boundaries
    # -----------------------------
    shapefile_path = r"C:\Users\于嘉诚\Desktop\STDM\Beijing location map\Beijing.shp"
    beijing = gpd.read_file(shapefile_path)
    lon_min, lat_min, lon_max, lat_max = beijing.total_bounds

    # -----------------------------
    # 4. Perform IDW interpolation
    # -----------------------------
    # Extract observation points and their corresponding AQI values
    points = df_merged[["lon", "lat"]].values
    values = df_merged["AQI"].values

    # Generate the interpolation grid
    grid_resolution = 500
    grid_x = np.linspace(lon_min, lon_max, grid_resolution)
    grid_y = np.linspace(lat_min, lat_max, grid_resolution)

    # meshgrid returns coordinate matrices of shape (len(grid_y), len(grid_x))
    mesh_lon, mesh_lat = np.meshgrid(grid_x, grid_y)

    # Initialize an array to store the interpolation results
    grid_idw = np.empty_like(mesh_lon, dtype=float)

    # Compute IDW interpolation for each grid point (naive implementation, may be slow)
    M, N = mesh_lon.shape
    for i in range(M):
        for j in range(N):
            x = mesh_lon[i, j]
            y = mesh_lat[i, j]
            grid_idw[i, j] = idw_naive(points, values, x, y, power=2)

    # -----------------------------
    # 5. Apply a mask: keep only the regions within the Beijing shapefile boundary
    # -----------------------------
    mask = contains(beijing.unary_union, mesh_lon, mesh_lat)
    grid_idw_masked = np.where(mask, grid_idw, np.nan)

    # -----------------------------
    # 6. Plot the results: save the interpolated map within the Beijing region
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the Beijing boundary
    beijing.boundary.plot(ax=ax, color='black', linewidth=1)

    # Plot the interpolation contour map
    contour = ax.contourf(mesh_lon, mesh_lat, grid_idw_masked, levels=50,
                          cmap='RdYlGn_r', alpha=0.8)

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=ax, label="AQI (IDW Interpolated)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # save the picture
    output_path = r"C:\Users\于嘉诚\Desktop\STDM\plot2022BJ.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
