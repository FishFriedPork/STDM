import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.gridspec as gridspec

def parse_station_excel(station_excel_path):
    """
    Reads the Excel file, automatically detects regions (in Chinese),
    maps them to English region names, and returns a DataFrame with columns:
    [station, lon, lat, region].
    """
    df_raw = pd.read_excel(station_excel_path, names=["Station", "Longitude", "Latitude"], header=0)

    region_map = {
        "城六区": "Six Urban Districts",
        "西北部": "Northwest",
        "东北部": "Northeast",
        "东南部": "Southeast",
        "西南部": "Southwest"
    }

    data_list = []
    current_region_cn = None

    for _, row in df_raw.iterrows():
        station_val = row["Station"]
        lon_val = row["Longitude"]
        lat_val = row["Latitude"]

        # If both longitude and latitude are NaN, treat this row as a region label
        if pd.isna(lon_val) and pd.isna(lat_val):
            current_region_cn = station_val
        else:
            if current_region_cn is None:
                raise ValueError("No region name found before station row. Check the Excel format.")
            # Map the Chinese region name to English
            region_en = region_map.get(current_region_cn, current_region_cn)
            data_list.append({
                "station": station_val,
                "lon": lon_val,
                "lat": lat_val,
                "region": region_en
            })

    df_station = pd.DataFrame(data_list)
    return df_station

def plot_regional_aqi_subplots(daily_aqi_csv, station_excel_path, output_png):
    """
    Plots the 5-day average AQI trends for five regions in subplots arranged in a 2x3 grid.
    - There are 6 subplot positions; only 5 regions are available, so the 6th subplot is hidden.
    - Each subplot displays the 5-day average AQI trend with the region name as the title.
    - The x-axis shows ticks every 6 months with labels rotated by 45°.
    - A margin is added to the x-axis limits so that the first and last data points do not touch the frame.
    - The final figure is saved as output_png.
    """
    # 1) Parse the station Excel file
    df_station = parse_station_excel(station_excel_path)

    # 2) Read the daily AQI CSV file (wide format: Date, Station1, Station2, ...)
    df_all = pd.read_csv(daily_aqi_csv, encoding='utf-8')
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="%Y-%m-%d")
    df_all = df_all.sort_values(by="Date")

    # 3) Convert wide format to long format: [Date, station, AQI]
    df_long = df_all.melt(id_vars="Date", var_name="station", value_name="AQI")

    # 4) Merge station region information
    df_merged = pd.merge(df_long, df_station[["station", "region"]], on="station", how="left")
    if df_merged["region"].isna().any():
        missing_stations = df_merged[df_merged["region"].isna()]["station"].unique()
        print("Warning: The following stations have no assigned region:", missing_stations)

    # 5) Group by [Date, region] and compute daily average AQI
    df_region_daily = df_merged.groupby(["Date", "region"])["AQI"].mean().reset_index()

    # 6) Pivot to wide format and resample to 5-day averages
    df_pivot = df_region_daily.pivot(index="Date", columns="region", values="AQI")
    df_pivot = df_pivot.resample('5D').mean()

    # Determine common x-axis limits based on the overall date range from df_pivot.
    # If df_pivot is empty, use a default range.
    if not df_pivot.empty:
        common_xlim = (df_pivot.index.min(), df_pivot.index.max())
    else:
        common_xlim = (pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31'))

    # Add a margin to the x-axis limits (5% of the total range on each side)
    total_range = common_xlim[1] - common_xlim[0]
    margin = total_range * 0.05
    new_xlim = (common_xlim[0] - margin, common_xlim[1] + margin)

    region_list = ["Six Urban Districts", "Northeast", "Northwest", "Southeast", "Southwest"]

    # 7) Create a 2x3 grid of subplots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axes_list = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    # Define a common date locator and formatter for all subplots
    locator = mdates.MonthLocator(interval=6)
    formatter = mdates.DateFormatter('%Y-%m')

    # Plot the 5 regions in the first 5 subplots
    for i, reg in enumerate(region_list):
        ax = axes_list[i]
        if reg in df_pivot.columns:
            ax.plot(df_pivot.index, df_pivot[reg], label=reg, color='#FFA500')
            ax.legend()
        else:
            # Display "No Data" text in the center if no data is available
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
            # For empty plots, force x-ticks using new_xlim
            tick_dates = pd.date_range(start=new_xlim[0], end=new_xlim[1], freq='6MS')
            ax.set_xticks(tick_dates)

        # Set the title and axis labels for every subplot
        ax.set_title(reg, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")

        # Apply the date locator and formatter to the x-axis
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        # Set the x-axis limits with margin so that data points do not touch the frame edges
        ax.set_xlim(new_xlim)

    # Hide the 6th subplot (unused)
    axes_list[5].set_visible(False)

    # Adjust layout to prevent overlap and to provide extra space at the bottom for x-axis labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.5, top=0.9)

    plt.savefig(output_png, dpi=300)
    plt.close()

def main():
    daily_aqi_csv = r"C:\Users\于嘉诚\Desktop\STDM\beijing_daily_avg_aqi_all_years.csv"
    station_excel_path = r"C:\Users\于嘉诚\Desktop\STDM\raw data\北京市站点列表-2021.01.23起.xlsx"
    output_png = r"C:\Users\于嘉诚\Desktop\STDM\timeplot.png"

    plot_regional_aqi_subplots(daily_aqi_csv, station_excel_path, output_png)

if __name__ == "__main__":
    main()
