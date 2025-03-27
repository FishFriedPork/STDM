# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran
from scipy.spatial import Voronoi
from shapely.geometry import Polygon


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Convert infinite regions in a 2D Voronoi diagram into finite polygons.
    Parameters:
        vor: a scipy.spatial.Voronoi object
        radius: distance used to extend infinite regions (default is automatically determined from the points' extent)
    Returns:
        regions: list of vertex indices for each region
        vertices: array of coordinates for all region vertices
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # Build a dictionary of all ridges for each point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Process each region corresponding to each point
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            # Finite region, use directly
            new_regions.append(vertices)
            continue

        # For infinite regions, reconstruct them
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            # v1 is an infinite vertex
            # Compute the tangent vector for the neighboring point
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            # Compute the normal vector
            n = np.array([-t[1], t[0]])
            # Find a sufficiently distant point
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort the region's vertices in counterclockwise order
        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [new_region[i] for i in np.argsort(angles)]
        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)


def compute_global_moran_aggregated():
    """
    Compute the Global Moran's I for aggregated AQI data at each station
    from January 1, 2022 to March 1, 2025.
    The value for each station is the mean of its daily AQI values over the period.
    The results are printed with three decimal places.
    """
    # -----------------------------
    # 1. Read daily average AQI data for all dates
    # -----------------------------
    aqi_file = r"C:\Users\于嘉诚\Desktop\STDM\beijing_daily_avg_aqi_all_years.csv"
    df_all = pd.read_csv(aqi_file, encoding='utf-8')
    print("Read AQI data: {} records".format(df_all.shape[0]))
    df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y-%m-%d')

    # Define the time range
    start_date = pd.to_datetime("20220101", format='%Y%m%d')
    end_date = pd.to_datetime("20250301", format='%Y%m%d')
    df_period = df_all[(df_all['Date'] >= start_date) & (df_all['Date'] <= end_date)]
    print("Filtered AQI data from {} to {}: {} records".format(start_date.date(), end_date.date(), df_period.shape[0]))
    if df_period.empty:
        print("No data found in the specified date range.")
        return

    # -----------------------------
    # 2. Compute the mean AQI for each station over the period
    # -----------------------------
    df_mean = df_period.drop(columns=["Date"]).mean().reset_index()
    df_mean.columns = ['station', 'AQI']
    print("Computed mean AQI for {} stations.".format(df_mean.shape[0]))

    # -----------------------------
    # 3. Read station latitude and longitude information and merge with AQI data
    # -----------------------------
    station_file = r"C:\Users\于嘉诚\Desktop\STDM\raw data\北京市站点列表-2021.01.23起.xlsx"
    df_station = pd.read_excel(station_file)
    print("Read station info: {} records".format(df_station.shape[0]))
    df_station.rename(columns={
        "监测点": "station",
        "经度": "lon",
        "纬度": "lat"
    }, inplace=True)

    df_merged = pd.merge(df_station, df_mean, on="station", how="inner")
    print("Merged data: {} records".format(df_merged.shape[0]))
    if df_merged.empty:
        print("No matching station data found. Please check if station names match.")
        return

    # -----------------------------
    # 4. Construct the Voronoi diagram and compute the spatial weights matrix
    # -----------------------------
    # Extract coordinates and AQI data
    coords = df_merged[['lon', 'lat']].values
    aqi_values = df_merged['AQI'].values

    # Generate Voronoi diagram
    vor = Voronoi(coords)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Construct polygons for each station
    polygons = [Polygon(vertices[region]) for region in regions]

    # Create a GeoDataFrame (CRS set to WGS84)
    gdf = gpd.GeoDataFrame(df_merged, geometry=polygons, crs="EPSG:4326")

    # Construct spatial weights matrix using the Queen contiguity rule
    w = Queen.from_dataframe(gdf)
    print("Spatial weights matrix constructed.")

    # -----------------------------
    # 5. Compute Global Moran's I
    # -----------------------------
    moran = Moran(aqi_values, w)
    print("Global Moran's I: {:.3f}".format(moran.I))
    print("p-value (normal approximation): {:.3f}".format(moran.p_norm))


if __name__ == "__main__":
    compute_global_moran_aggregated()
