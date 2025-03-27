# -*- coding: utf-8 -*-

import os
import re
import pandas as pd

def calculate_daily_avg_aqi_multiple_folders(input_folders, output_file):
    """
    Reads CSV files from multiple folders whose filenames match the pattern "beijing_all_YYYYMMDD.csv",
    filters rows where 'type' equals 'AQI', computes the daily average for each station (ignoring missing values),
    aggregates all daily averages, and writes the combined data to a specified CSV file.

    New requirements:
      1. For each station, use the average of hourly AQI data as the daily AQI.
      2. For missing data, fill with the previous day's value.
      3. For data that exceed 3 times or fall below 30% of the station's overall daily average,
         treat them as anomalies and replace with the previous day's value.
    """
    all_daily_averages = []  # List to store daily AQI averages for each day
    pattern = re.compile(r"^beijing_all_\d{8}\.csv$")  # Regular expression pattern to match file names

    for folder in input_folders:
        for file_name in os.listdir(folder):
            if not pattern.match(file_name):
                continue

            file_path = os.path.join(folder, file_name)
            try:
                # Read CSV data and drop rows that are completely empty
                df = pd.read_csv(file_path, encoding='utf-8').dropna(how='all')
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            # Filter rows where the 'type' column equals 'AQI' and drop rows that are entirely empty
            df_aqi = df[df['type'] == 'AQI'].dropna(how='all')
            if df_aqi.empty:
                print(f"No valid AQI data in file {file_path}, skipping.")
                continue

            # Assume that the first three columns are [date, hour, type] and the remaining columns contain station data
            station_cols = df_aqi.columns[3:]
            if len(station_cols) == 0:
                print(f"No station data columns in file {file_path}, skipping.")
                continue

            # Calculate the average for each station (ignoring missing values)
            daily_avg = df_aqi[station_cols].mean()

            # Extract the date from the file name (format "YYYYMMDD")
            date_str = file_name.replace("beijing_all_", "").replace(".csv", "")
            daily_avg.name = date_str

            all_daily_averages.append(daily_avg)

    # Combine all daily averages into a single DataFrame
    if all_daily_averages:
        result_df = pd.DataFrame(all_daily_averages)
        result_df.index.name = 'Date'
        # Convert the index (date string) to datetime for proper sorting and processing
        result_df.index = pd.to_datetime(result_df.index, format="%Y%m%d")
        result_df = result_df.sort_index()

        # 1. Fill missing data with previous day's value
        result_df = result_df.fillna(method='ffill')

        # 2. For each station, compute the overall mean and replace anomalous values:
        #    if a day's value > 3 * overall mean or < 0.3 * overall mean,
        #    replace it with the previous day's value.
        for col in result_df.columns:
            overall_mean = result_df[col].mean()
            for i in range(1, len(result_df)):
                value = result_df.iloc[i][col]
                if value > 3 * overall_mean or value < 0.3 * overall_mean:
                    result_df.iloc[i, result_df.columns.get_loc(col)] = result_df.iloc[i-1][col]

        # Write the processed data to a CSV file
        result_df.to_csv(output_file, encoding='utf-8')
        print(f"Calculation complete. Results saved to {output_file}")
    else:
        print("No valid AQI data found.")

if __name__ == "__main__":
    # Define the root directory containing the raw data folders
    raw_data_dir = r"C:\Users\于嘉诚\Desktop\STDM\raw data"
    # List all subdirectories in raw_data_dir that start with "beijing_"
    input_folders = [
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d)) and d.startswith("beijing_")
    ]

    # Define the output file path (saving in the STDM folder)
    output_file = r"C:\Users\于嘉诚\Desktop\STDM\beijing_daily_avg_aqi_all_years.csv"

    calculate_daily_avg_aqi_multiple_folders(input_folders, output_file)
