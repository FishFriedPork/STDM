# ------------------------------
# 0. Load required packages
# ------------------------------
library(readxl)      # Read Excel files
library(dplyr)       # Data manipulation
library(geosphere)   # Calculate geographic distances
library(spdep)       # Construct spatial weight matrices
library(ggplot2)     # Plotting
library(writexl)     # Export Excel files
library(deldir)      # Generate Voronoi diagrams
library(sp)          # Process spatial polygons
library(lubridate)   # Handle dates
library(tidyr)       # Reshape data
Sys.setlocale("LC_TIME", "C")

# ------------------------------
# 1. Read and process AQI data (daily data)
# ------------------------------
aqi_file <- "C:/Users/于嘉诚/Desktop/STDM/beijing_daily_avg_aqi_all_years.csv"
aqi_data <- read.csv(aqi_file, header = TRUE, sep = ",", 
                     stringsAsFactors = FALSE, fileEncoding = "UTF-8")
cat("Original AQI data dimensions:", dim(aqi_data), "\n")
aqi_data$Date <- as.Date(aqi_data$Date, format = "%Y-%m-%d")
aqi_data <- aqi_data %>% filter(!is.na(Date)) %>% arrange(Date)
row.names(aqi_data) <- paste(aqi_data$Date, seq_len(nrow(aqi_data)), sep = "_")

# Construct the AQI matrix (excluding the Date column; rows represent time, columns represent stations)
aqi_matrix <- data.matrix(aqi_data[, !(names(aqi_data) %in% c("Date"))])
cat("Processed AQI data - rows:", nrow(aqi_matrix), "; columns:", ncol(aqi_matrix), "\n")
station_names <- colnames(aqi_matrix)

# ------------------------------
# 2. Read and process station geographic information
# ------------------------------
coords_file <- "C:/Users/于嘉诚/Desktop/STDM/raw data/北京市站点列表-2021.01.23起.xlsx"
coords_data <- read_xlsx(coords_file)
colnames(coords_data) <- c("Station", "Longitude", "Latitude")
# Align data based on station_names
coords_data <- coords_data[match(station_names, coords_data$Station), ]
missing_stations <- station_names[is.na(coords_data$Longitude) | is.na(coords_data$Latitude)]
if(length(missing_stations) > 0) {
  cat("Warning: The following stations are missing coordinates:", 
      paste(missing_stations, collapse = ", "), "\n")
  valid_idx <- which(!(station_names %in% missing_stations))
  coords_data <- coords_data[valid_idx, ]
  station_names <- station_names[valid_idx]
  aqi_matrix <- aqi_matrix[, valid_idx]
}
longitude_vec <- coords_data$Longitude
latitude_vec  <- coords_data$Latitude
N <- length(station_names)
cat("Number of matched stations (after removing missing):", N, "\n")

# ------------------------------
# 3. Construct spatial weight matrix using Voronoi diagram (Queen contiguity)
# ------------------------------
voronoi_result <- deldir(longitude_vec, latitude_vec)
tiles <- tile.list(voronoi_result)
polys <- vector("list", length(tiles))
for(i in 1:length(tiles)){
  coords_tile <- cbind(tiles[[i]]$x, tiles[[i]]$y)
  coords_tile <- rbind(coords_tile, coords_tile[1,])  # Close the polygon
  polys[[i]] <- Polygons(list(Polygon(coords_tile)), ID = as.character(i))
}
sp_polys <- SpatialPolygons(polys)
nb <- poly2nb(sp_polys, queen = TRUE)
Wlistw <- nb2listw(nb, style = "W", zero.policy = TRUE)
Wmat <- listw2mat(Wlistw)
cat("Spatial weight matrix based on Voronoi diagram has been constructed.\n")

# ------------------------------
# 4. Split data into training and testing sets (90%/10%)
# ------------------------------
n_obs <- nrow(aqi_matrix)
train_size <- floor(n_obs * 0.9)
test_size  <- n_obs - train_size
train_matrix <- aqi_matrix[1:train_size, ]
test_matrix  <- aqi_matrix[(train_size + 1):n_obs, ]
cat("Training set rows:", nrow(train_matrix), "; Testing set rows:", nrow(test_matrix), "\n")

# ------------------------------
# 5. STARIMA model fitting (set p=2, q=2, seasonal differencing d = 365; for daily data)
# ------------------------------
# Note: Here, d is set to 365 (for daily data). Adjust as necessary.
source("C:/Users/于嘉诚/Desktop/STDM/Data/starima_package.R")
p <- 2
q <- 2
d <- 365
fit.star <- starima_fit(Z = train_matrix, W = list(w1 = Wmat), p = p, d = d, q = q)
cat("STARIMA model fitting is complete.\n")

# ------------------------------
# 6. Testing set prediction and evaluation (calculate evaluation metrics per station)
# ------------------------------
# To ensure sufficient initial conditions for prediction, set offset = d + (p + q)
offset <- d + (p + q)  # 365 + 4 = 369
start_index <- train_size - offset + 1
data_for_pre_test <- aqi_matrix[start_index:n_obs, ]
pre.star_test <- starima_pre(data_for_pre_test, model = fit.star)
# Assume the returned prediction matrix pre.star_test$PRE includes initial conditions and predictions.
# Take the last test_size rows as the testing set prediction results.
predicted_test <- pre.star_test$PRE
predicted_test <- predicted_test[(nrow(predicted_test) - test_size + 1):nrow(predicted_test), ]

# Calculate evaluation metrics for each station (comparing predicted and actual values on the testing set)
station_eval <- sapply(1:ncol(test_matrix), function(j) {
  actual <- test_matrix[, j]
  pred   <- predicted_test[, j]
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  r2   <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  c(RMSE = rmse, MAE = mae, R2 = r2)
})
station_eval <- as.data.frame(t(station_eval))
station_eval <- cbind(Station = rownames(station_eval), station_eval)

# Also calculate evaluation metrics for the overall (average of all stations)
avg_actual_test <- rowMeans(test_matrix)
avg_predicted_test <- rowMeans(predicted_test)
RMSE_avg <- sqrt(mean((avg_actual_test - avg_predicted_test)^2))
MAE_avg  <- mean(abs(avg_actual_test - avg_predicted_test))
R2_avg   <- 1 - sum((avg_actual_test - avg_predicted_test)^2) / sum((avg_actual_test - mean(avg_actual_test))^2)
overall_eval <- data.frame(Station = "ALL",
                           RMSE = RMSE_avg, MAE = MAE_avg, R2 = R2_avg)

# Combine station and overall evaluation metrics
eval_df <- rbind(station_eval, overall_eval)

# Plot the daily average AQI for the testing set (average over all stations)
# Actual values are shown as a solid black line and predicted values as a dashed black line.
test_dates <- aqi_data$Date[(train_size + 1):n_obs]
avg_df <- data.frame(Date = test_dates,
                     Actual = avg_actual_test,
                     Predicted = avg_predicted_test)
# Reshape data to long format to enable a legend
avg_df_long <- pivot_longer(avg_df, cols = c("Actual", "Predicted"),
                            names_to = "Type", values_to = "AQI")
p_starima <- ggplot(avg_df_long, aes(x = Date, y = AQI, color = Type, linetype = Type)) +
  geom_line(size = 1) +
  labs(x = "Date", y = "Daily Average AQI") +  # No title provided
  scale_x_date(date_labels = "%Y-%m-%d", date_breaks = "1 month") +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "black")) +
  scale_linetype_manual(values = c("Actual" = "solid", "Predicted" = "dashed")) +
  theme_minimal() +
  # Add black solid border lines on the top and right, and set legend text in black
  theme(
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.text = element_text(color = "black"),
    legend.text = element_text(color = "black"),
    legend.title = element_blank(),
    legend.background = element_rect(color = "black", fill = NA),
    legend.position = c(1, 1),         # Positions the legend at the top right
    legend.justification = c(1, 1),      # Ensures the legend's top right corner is at the (1,1) point
    panel.grid = element_blank()         # Remove grid lines
  )
# Save the testing set prediction plot
output_plot_path <- "C:/Users/于嘉诚/Desktop/STDM/Fig_STARIMA.png"
ggsave(filename = output_plot_path, plot = p_starima, width = 10, height = 6)
cat("The plot of actual vs predicted values on the testing set has been saved to:", output_plot_path, "\n")

# ------------------------------
# 7. Future Prediction: Predict data from 2025/03/02 to 2025/06/01 (saved per station)
# ------------------------------
forecast_start <- as.Date("2025-03-02")
forecast_end   <- as.Date("2025-06-01")
n_forecast <- as.numeric(forecast_end - forecast_start) + 1  # Number of days to predict
# Construct initial conditions for future prediction:
# Use the last offset rows of historical data, then append n_forecast rows of placeholder data (filled with the last day's values)
future_placeholder <- matrix(rep(aqi_matrix[n_obs, ], times = n_forecast),
                             nrow = n_forecast, byrow = TRUE)
init_future <- rbind(aqi_matrix[(n_obs - offset + 1):n_obs, ], future_placeholder)
pre.star_future <- starima_pre(init_future, model = fit.star)
# Assume the returned prediction matrix pre.star_future$PRE includes initial conditions and predictions.
# Extract the future prediction part, i.e., the last n_forecast rows (all stations)
predicted_future <- pre.star_future$PRE
predicted_future <- predicted_future[(nrow(predicted_future) - n_forecast + 1):nrow(predicted_future), ]

# Construct future prediction dates
forecast_dates <- seq.Date(from = forecast_start, to = forecast_end, by = "day")
# Construct a prediction dataframe: the first column is Date and subsequent columns are predictions for each station
future_pred_df <- as.data.frame(predicted_future)
colnames(future_pred_df) <- station_names
future_pred_df <- cbind(Date = forecast_dates, future_pred_df)

# ------------------------------
# 8. Save future prediction results and evaluation metrics in one Excel file
# ------------------------------
output_excel_path <- "C:/Users/于嘉诚/Desktop/STDM/STARIMA_Predict.xlsx"
# Use writexl to write multiple sheets; here we create a list of data frames
write_xlsx(list("Predictions" = future_pred_df,
                "Evaluation"  = eval_df),
           path = output_excel_path)
cat("Future prediction data (per station) and testing set evaluation metrics have been saved to:", output_excel_path, "\n")
