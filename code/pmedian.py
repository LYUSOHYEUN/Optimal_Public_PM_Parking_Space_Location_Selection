import pandas as pd
import numpy as np
from pyproj import Proj, transform
from haversine import haversine

# File paths
DATA_PATH = "data/"
OUTPUT_PATH = "output/"
CENTER_FILE = DATA_PATH + "center.csv"
BUS_FILE = DATA_PATH + "bus.csv"
SUBWAY_FILE = DATA_PATH + "subway.xlsx"
PARK_FILE = DATA_PATH + "park.csv"
TOW_FILE = DATA_PATH + "tow.csv"

# Load data
center = pd.read_csv(CENTER_FILE)
bus = pd.read_csv(BUS_FILE, encoding='cp949')
subway = pd.read_excel(SUBWAY_FILE)
park = pd.read_csv(PARK_FILE, encoding='cp949')
tow = pd.read_csv(TOW_FILE, encoding='cp949')

# Data preprocessing
center['X'] = (center['left'] + center['right']) / 2
center['Y'] = (center['bottom'] + center['top']) / 2
center.dropna(inplace=True)

# Coordinate transformation
original_crs = Proj(init='epsg:5179')  # UTM-K
target_crs = Proj(init='epsg:4326')   # WGS84

def convert_coordinates(x, y):
    lon, lat = transform(original_crs, target_crs, x, y)
    return lat, lon

center['경도'], center['위도'] = zip(*center.apply(lambda row: convert_coordinates(row['X'], row['Y']), axis=1))
center.rename(columns={'위도': '경도', '경도': '위도'}, inplace=True)

# Convert data to points
center_points = np.array(list(zip(center['위도'], center['경도'])))
bus_points = np.array(list(zip(bus['위도'], bus['경도'])))
subway_points = np.array(list(zip(subway['위도'], subway['경도'])))
park_points = np.array(list(zip(park['위도'], park['경도'])))
tow_points = np.array(list(zip(tow['위도'], tow['경도'])))

# Calculate weights
total_w = len(bus_points) + len(subway_points) + len(park_points) + len(tow_points)
weights = {
    'bus': (total_w - len(bus_points)) / total_w,
    'subway': (total_w - len(subway_points)) / total_w,
    'park': (total_w - len(park_points)) / total_w,
    'tow': (total_w - len(tow_points)) / total_w,
}

# P-median calculation function
def calculate_pmedian(center_points, facility_points, weight):
    distances = []
    for center in center_points:
        distances.append([haversine(center, facility) * 1000 for facility in facility_points])
    min_distances = np.min(distances, axis=1)
    weighted_distances = np.where(distances == min_distances[:, None], weight, 0)
    return np.sum(weighted_distances, axis=0)

D1 = calculate_pmedian(center_points, bus_points, weights['bus'])
D2 = calculate_pmedian(center_points, subway_points, weights['subway'])
D3 = calculate_pmedian(center_points, park_points, weights['park'])
D4 = calculate_pmedian(center_points, tow_points, weights['tow'])

D_final = D1 + D2 + D3 + D4
result_df = pd.DataFrame(D_final, columns=['Weight']).sort_values(by='Weight', ascending=False)

# Save results
result_df_30 = result_df.head(30)
result_df_30.to_csv(OUTPUT_PATH + "P-median_30.csv", index=False)