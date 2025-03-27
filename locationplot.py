# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


def plot_stations_on_beijing_map(shapefile_path, station_file, output_path):
    """
    Plot the locations of monitoring stations on the Beijing map and save the output image.

    :param shapefile_path: str, path to the Beijing shapefile
    :param station_file: str, path to the Excel file containing station latitude and longitude data
    :param output_path: str, path to save the output image (including file name)
    """
    # 1. Read the Beijing shapefile
    beijing = gpd.read_file(shapefile_path)

    # 2. Read the station data
    #    The Excel file is assumed to contain the columns: "监测点" (station), "经度" (longitude), "纬度" (latitude)
    df_station = pd.read_excel(station_file)
    df_station.rename(columns={
        "监测点": "station",
        "经度": "lon",
        "纬度": "lat"
    }, inplace=True)

    # 3. Convert the station data into a GeoDataFrame
    gdf_station = gpd.GeoDataFrame(
        df_station,
        geometry=[Point(xy) for xy in
                  zip(df_station["lon"], df_station["lat"])],
        crs="EPSG:4326"  # WGS84 coordinate system
    )

    # 4. Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the Beijing map with a light earth-tone fill (tan) and black edges
    beijing.plot(ax=ax, color='tan', edgecolor='black', linewidth=1)

    # Plot the station locations in black with larger markers
    gdf_station.plot(ax=ax, marker='o', color='black', markersize=50,
                     label='Stations')

    # Set labels (no title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    # 5. Save the output image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    shapefile_path = r"C:\Users\于嘉诚\Desktop\STDM\Beijing location map\Beijing.shp"
    station_file = r"C:\Users\于嘉诚\Desktop\STDM\raw data\北京市站点列表-2021.01.23起.xlsx"
    output_path = r"C:\Users\于嘉诚\Desktop\STDM\locplot.png"

    plot_stations_on_beijing_map(shapefile_path, station_file, output_path)
