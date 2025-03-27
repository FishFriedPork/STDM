# -------------------------------------
# Load Required Libraries
# -------------------------------------
library(readxl)    # Read Excel files
library(dplyr)     # Data manipulation
library(geosphere) # Calculate geographic distances
library(spdep)     # Construct spatial weight matrices
library(ggplot2)   # Plotting
library(writexl)   # Export to Excel
library(deldir)    # Generate Voronoi diagrams
library(sp)        # Handle spatial polygons

# -------------------------------------
# 1. Read and Process AQI Data (Daily Data, No Monthly Aggregation)
# -------------------------------------
aqi_file <- "C:/Users/于嘉诚/Desktop/STDM/beijing_daily_avg_aqi_all_years.csv"
aqi_data <- read.csv(aqi_file,
                     header = TRUE,
                     sep = ",",
                     stringsAsFactors = FALSE,
                     fileEncoding = "UTF-8")
cat("Original AQI data dimensions:", dim(aqi_data), "\n")
aqi_data$Date <- as.Date(aqi_data$Date, format = "%Y-%m-%d")
aqi_data <- aqi_data %>% filter(!is.na(Date)) %>% arrange(Date)
row.names(aqi_data) <- paste(aqi_data$Date, seq_len(nrow(aqi_data)), sep = "_")

# Construct the daily data matrix by excluding the Date column
aqi_matrix <- data.matrix(aqi_data[, !(names(aqi_data) %in% c("Date"))])
cat("Processed daily average AQI data - rows:", nrow(aqi_matrix), "; columns:", ncol(aqi_matrix), "\n")
station_names <- colnames(aqi_matrix)

# -------------------------------------
# 2. Read and Process Station Geographic Information
# -------------------------------------
coords_file <- "C:/Users/于嘉诚/Desktop/STDM/raw data/北京市站点列表-2021.01.23起.xlsx"
coords_data <- read_xlsx(coords_file)
colnames(coords_data) <- c("Station", "Longitude", "Latitude")

# Reorder according to station_names
coords_data <- coords_data[match(station_names, coords_data$Station), ]

# Identify stations with missing coordinates and update station_names and aqi_matrix
missing_stations <- station_names[is.na(coords_data$Longitude) | is.na(coords_data$Latitude)]
if(length(missing_stations) > 0) {
  cat("Warning: The following stations have missing coordinates:", paste(missing_stations, collapse = ", "), "\n")
  valid_idx <- which(!(station_names %in% missing_stations))
  coords_data <- coords_data[valid_idx, ]
  station_names <- station_names[valid_idx]
  aqi_matrix <- aqi_matrix[, valid_idx]
}
longitude_vec <- coords_data$Longitude
latitude_vec  <- coords_data$Latitude
N <- length(station_names)
cat("Number of matched stations (after removing missing values):", N, "\n")

# -------------------------------------
# 3. Construct Spatial Weight Matrix Using Voronoi Diagram (Queen Contiguity)
# -------------------------------------
voronoi_result <- deldir(longitude_vec, latitude_vec)
tiles <- tile.list(voronoi_result)
polys <- vector("list", length(tiles))
for(i in 1:length(tiles)){
  coords_tile <- cbind(tiles[[i]]$x, tiles[[i]]$y)
  coords_tile <- rbind(coords_tile, coords_tile[1,])
  polys[[i]] <- Polygons(list(Polygon(coords_tile)), ID = as.character(i))
}
sp_polys <- SpatialPolygons(polys)
nb <- poly2nb(sp_polys, queen = TRUE)
Wlistw <- nb2listw(nb, style = "W", zero.policy = TRUE)
Wmat <- listw2mat(Wlistw)
cat("Spatial weight matrix based on Voronoi diagram has been constructed.\n")

# -------------------------------------
# 4. Split Data into Training and Testing Sets (90% / 10%)
# -------------------------------------
n_obs <- nrow(aqi_matrix)
train_size <- floor(n_obs * 0.9)
train_matrix <- aqi_matrix[1:train_size, ]
test_matrix  <- aqi_matrix[(train_size + 1):n_obs, ]
cat("Number of training samples:", nrow(train_matrix), "; Number of testing samples:", nrow(test_matrix), "\n")

# -------------------------------------
# 4.1 Apply a 365-Difference to the Training Data
# -------------------------------------
diff_train_matrix <- diff(train_matrix, lag = 365)
cat("Number of training samples after applying 365-difference:", nrow(diff_train_matrix), "\n")

# -------------------------------------
# 5. Calculate and Plot STACF and STPACF for the 365-Differenced Data
# -------------------------------------
source("C:/Users/于嘉诚/Desktop/STDM/Data/starima_package.R")
stacf_result <- stacf(diff_train_matrix, Wmat, nLags = 50)
cat("STACF result based on 365-differenced data:\n")
print(stacf_result$stacf)

stpacf_result <- stpacf(diff_train_matrix, Wmat)
cat("STPACF result based on 365-differenced data:\n")
print(stpacf_result$stpacf)
