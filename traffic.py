# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set a visually appealing style for the plots
sns.set(style="whitegrid")
plt.style.use('fivethirtyeight')

# --- Define Constants for Plot Labels ---
Y_LABEL_ACCIDENTS = 'Number of Accidents'

print("Libraries imported successfully.")

# Step 2: Load and Prepare the Data
try:
    # The dataset file should be in the same folder as the script
    file_path = 'US_Accidents_March23.csv'
    print(f"Loading data from '{file_path}'...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print("Dataset loaded successfully.")
    
    # --- Data Sampling ---
    # The dataset is very large. We'll take a random sample to make analysis faster.
    # We use a random_state for reproducibility.
    sample_size = 100000
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"Using a random sample of {sample_size} records for analysis.")

except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please make sure the 'US_Accidents_March23.csv' file is in the same folder as this script.")
    exit()

# Step 3: Data Cleaning and Preprocessing
print("\nStarting data cleaning and preprocessing...")

# Select a subset of columns relevant to the task to save memory
columns_to_keep = [
    'Start_Time', 'Start_Lat', 'Start_Lng', 'Severity', 
    'Weather_Condition', 'Sunrise_Sunset', 'Junction', 'Crossing', 'Traffic_Signal'
]
df_clean = df_sample[columns_to_keep].copy()

# Drop rows with missing values in key columns
df_clean.dropna(subset=['Start_Time', 'Start_Lat', 'Start_Lng', 'Weather_Condition', 'Sunrise_Sunset'], inplace=True)

# Convert 'Start_Time' to datetime objects for time-based analysis
df_clean['Start_Time'] = pd.to_datetime(df_clean['Start_Time'], errors='coerce')

# Extract useful time-based features
df_clean['Hour'] = df_clean['Start_Time'].dt.hour
df_clean['DayOfWeek'] = df_clean['Start_Time'].dt.day_name()

print("Data cleaning and preprocessing complete.")


# Step 4: Exploratory Data Analysis (EDA) and Visualization

# --- Analysis 1: Accidents by Time of Day ---
plt.figure(figsize=(12, 6))
sns.countplot(x='Hour', data=df_clean, palette='viridis')
plt.title('Accidents by Hour of the Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel(Y_LABEL_ACCIDENTS, fontsize=12)
plt.show()

# --- Analysis 2: Accidents by Day of the Week ---
plt.figure(figsize=(10, 6))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(x='DayOfWeek', data=df_clean, order=day_order, palette='magma')
plt.title('Accidents by Day of the Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel(Y_LABEL_ACCIDENTS, fontsize=12)
plt.xticks(rotation=45)
plt.show()

# --- Analysis 3: Accidents by Weather Condition ---
plt.figure(figsize=(12, 8))
top_weather = df_clean['Weather_Condition'].value_counts().nlargest(10)
sns.barplot(y=top_weather.index, x=top_weather.values, palette='plasma', orient='h')
plt.title('Top 10 Weather Conditions During Accidents', fontsize=16)
plt.xlabel(Y_LABEL_ACCIDENTS, fontsize=12)
plt.ylabel('Weather Condition', fontsize=12)
plt.show()

# --- Analysis 4: Accidents by Time of Day (Day vs. Night) ---
plt.figure(figsize=(8, 6))
sns.countplot(x='Sunrise_Sunset', data=df_clean, palette='twilight')
plt.title('Accidents: Day vs. Night', fontsize=16)
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel(Y_LABEL_ACCIDENTS, fontsize=12)
plt.show()

# --- Analysis 5: Accidents by Road Feature ---
# This new visualization shows the impact of road features on accident frequency.
road_features = ['Junction', 'Crossing', 'Traffic_Signal']
road_features_df = df_clean[road_features].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=road_features_df.values, y=road_features_df.index, palette='cubehelix', orient='h')
plt.title('Accidents by Road Feature', fontsize=16)
plt.xlabel(Y_LABEL_ACCIDENTS, fontsize=12)
plt.ylabel('Road Feature', fontsize=12)
plt.show()


# Step 6: Visualize Accident Hotspots on a Map
print("\nGenerating accident hotspot map... (This may take a moment)")

# Create a base map centered on the US
# The location is an approximate center of the contiguous US
map_center = [39.8283, -98.5795]
accident_map = folium.Map(location=map_center, zoom_start=4)

# Create a heatmap layer
# Use a smaller subset for the heatmap to ensure it renders quickly
heat_data = df_clean[['Start_Lat', 'Start_Lng']].dropna().values.tolist()
HeatMap(heat_data, radius=8, max_zoom=13).add_to(accident_map)

# Save the map to an HTML file
map_filename = 'accident_hotspots.html'
accident_map.save(map_filename)

print(f"Map saved successfully as '{map_filename}'. Open this file in a web browser to view the heatmap.")