import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# NOAA CDO API Parameters
API_TOKEN = "FRwWABHmWvKUFavDEXitKEtYRqwYxZjm"  # Get this from https://www.ncdc.noaa.gov/cdo-web/token
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"


def fetch_station_data(station_id, start_date, end_date, datasetid="GHCND"):
    """
    Fetch weather data for a specific station using NOAA's CDO API.

    Args:
        station_id: NOAA station identifier (e.g., 'GHCND:USW00023234')
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        datasetid: Dataset identifier (default: GHCND - Daily Summaries)

    Returns:
        Pandas DataFrame with weather data
    """
    headers = {"token": API_TOKEN}

    # Parameters for the API request
    params = {
        "datasetid": datasetid,
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "units": "metric"
    }

    all_data = []
    offset = 0

    # CDO API has a pagination limit of 1000 records, so we need to loop
    while True:
        # Add offset to parameters
        params["offset"] = offset

        # Make API request
        print(f"Fetching data for {station_id}, offset {offset}...")
        response = requests.get(BASE_URL, headers=headers, params=params)

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break

        # Parse response
        data = response.json()

        # Check if there are any results
        if "results" not in data or len(data["results"]) == 0:
            break

        # Add results to our list
        all_data.extend(data["results"])

        # Update offset for next page
        offset += len(data["results"])

        # Check if we've received all data
        if len(data["results"]) < 1000:
            break

        # Respect API rate limits
        time.sleep(0.2)

    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    else:
        print(f"No data found for {station_id} between {start_date} and {end_date}")
        return pd.DataFrame()


def download_weather_data():
    """Download weather data for multiple stations over a time period."""

    # Define stations to use (these are example station IDs with the GHCND prefix)
    # You'll need to find valid station IDs from NOAA's station database
    stations = [
        "GHCND:USW00023234",  # Atlanta, GA
        "GHCND:USW00014739",  # New York, NY
        "GHCND:USW00023174",  # Miami, FL
        "GHCND:USW00024233",  # Seattle, WA
        "GHCND:USW00023188"  # Los Angeles, CA
    ]

    # Define date range (5 years of data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    # For each station, fetch data in smaller chunks to avoid API limits
    # NOAA's CDO API has limits on how much data you can retrieve in one request
    all_station_data = []

    for station in stations:
        station_df = pd.DataFrame()

        # Split into yearly chunks
        current_start = start_date
        while current_start < end_date:
            # Calculate chunk end date (1 year later or end_date, whichever comes first)
            current_end = min(
                (datetime.strptime(current_start, "%Y-%m-%d") + timedelta(days=365)).strftime("%Y-%m-%d"),
                end_date
            )

            # Fetch data for this chunk
            chunk_df = fetch_station_data(station, current_start, current_end)

            if not chunk_df.empty:
                station_df = pd.concat([station_df, chunk_df])

            # Update start date for next chunk
            current_start = (datetime.strptime(current_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

            # Save checkpoint for each station-year
            if not chunk_df.empty:
                year = datetime.strptime(current_end, "%Y-%m-%d").year
                chunk_df.to_csv(f"data/raw/{station.replace(':', '_')}_{year}.csv", index=False)

        if not station_df.empty:
            all_station_data.append(station_df)
            print(f"Completed download for {station}: {len(station_df)} records")

    # Combine all data
    if all_station_data:
        combined_df = pd.concat(all_station_data, ignore_index=True)
        combined_df.to_csv("data/raw/combined_weather_data.csv", index=False)
        print(f"Saved combined data with {len(combined_df)} records")
        return combined_df
    else:
        print("No data was downloaded. Check your API token and station IDs.")
        return pd.DataFrame()


def process_weather_data(df):
    """Process the raw weather data for use in the LSTM model."""

    if df.empty:
        print("No data to process")
        return df

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Identify the weather elements we're interested in
    # Common data types in GHCND: TMAX, TMIN, PRCP, SNOW, SNWD
    elements_of_interest = ['TMAX', 'TMIN', 'PRCP']

    # Filter to keep only the data types we need
    df = df[df['datatype'].isin(elements_of_interest)]

    # Pivot the data to have one row per station-date combination
    # with columns for each weather element
    pivoted_df = df.pivot_table(
        index=['station', 'date'],
        columns='datatype',
        values='value'
    ).reset_index()

    # Handle missing values
    for element in elements_of_interest:
        if element in pivoted_df.columns:
            # Fill missing values by station using forward fill then backward fill
            pivoted_df[element] = pivoted_df.groupby('station')[element].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )

    # Add derived features
    if 'TMAX' in pivoted_df.columns and 'TMIN' in pivoted_df.columns:
        pivoted_df['TEMP_RANGE'] = pivoted_df['TMAX'] - pivoted_df['TMIN']

    # Add time-based features
    pivoted_df['MONTH'] = pivoted_df['date'].dt.month
    pivoted_df['DAY_OF_YEAR'] = pivoted_df['date'].dt.dayofyear

    # Save processed data
    pivoted_df.to_csv("data/processed/processed_weather_data.csv", index=False)
    print(f"Saved processed data with {len(pivoted_df)} records")

    return pivoted_df


def create_time_series_features(df, window_size=7, target_col='TMAX', forecast_horizon=1):
    """Create time series features for LSTM model."""

    if df.empty:
        print("No data to create features from")
        return np.array([]), np.array([]), []

    # Sort by station and date
    df = df.sort_values(by=['station', 'date'])

    features = []
    targets = []
    stations = []

    # Feature columns
    feature_cols = ['TMAX', 'TMIN', 'PRCP']
    if 'TEMP_RANGE' in df.columns:
        feature_cols.append('TEMP_RANGE')
    if 'MONTH' in df.columns:
        feature_cols.append('MONTH')
    if 'DAY_OF_YEAR' in df.columns:
        feature_cols.append('DAY_OF_YEAR')

    # Group by station
    for station, station_df in df.groupby('station'):
        # Create sequences for each station
        for i in range(len(station_df) - window_size - forecast_horizon + 1):
            # Get window of data
            window = station_df.iloc[i:i + window_size]

            # Get target value
            target_idx = i + window_size + forecast_horizon - 1
            if target_idx < len(station_df):
                # Check if target column exists
                if target_col in station_df.columns:
                    target = station_df.iloc[target_idx][target_col]

                    # Check if we have all features for this window
                    if all(col in window.columns for col in feature_cols):
                        window_features = window[feature_cols].values

                        features.append(window_features)
                        targets.append(target)
                        stations.append(station)

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(targets)

    # Save features and targets
    np.save('data/processed/X_features.npy', X)
    np.save('data/processed/y_targets.npy', y)
    pd.DataFrame({'station': stations}).to_csv('data/processed/stations.csv', index=False)

    print(f"Created {len(X)} sequences with shape {X.shape}")

    return X, y, stations


def create_non_iid_splits(X, y, stations, num_clients=5):
    """Split data by station to create non-IID distribution for federated learning."""

    if len(X) == 0:
        print("No data to split")
        return []

    # Convert stations to numpy array if needed
    stations = np.array(stations)

    # Get unique stations
    unique_stations = np.unique(stations)

    # Ensure we have enough stations
    if len(unique_stations) < num_clients:
        print(f"Warning: Only {len(unique_stations)} stations available for {num_clients} clients")
        num_clients = len(unique_stations)

    # Create client data splits
    client_data = []

    # Distribute stations to clients
    stations_per_client = max(1, len(unique_stations) // num_clients)
    remainder = len(unique_stations) % num_clients

    start_idx = 0
    for client_idx in range(num_clients):
        # Calculate how many stations this client gets
        n_stations = stations_per_client + (1 if client_idx < remainder else 0)

        # If we've distributed all stations, break
        if start_idx >= len(unique_stations):
            break

        # Get stations for this client
        client_stations = unique_stations[start_idx:min(start_idx + n_stations, len(unique_stations))]
        start_idx += n_stations

        # Get indices for this client's data
        client_indices = np.isin(stations, client_stations)
        client_X = X[client_indices]
        client_y = y[client_indices]

        client_data.append((client_X, client_y))

        print(f"Client {client_idx}: {len(client_X)} samples from stations {client_stations}")

    # Save client data
    os.makedirs('data/processed/clients', exist_ok=True)
    for i, (client_X, client_y) in enumerate(client_data):
        np.save(f'data/processed/clients/client_{i}_X.npy', client_X)
        np.save(f'data/processed/clients/client_{i}_y.npy', client_y)

    return client_data


def visualize_data_distribution(client_data):
    """Visualize data distribution across clients."""

    if not client_data:
        print("No client data to visualize")
        return

    plt.figure(figsize=(12, 10))

    # Plot 1: Sample counts per client
    plt.subplot(2, 1, 1)
    client_sizes = [len(client_X) for client_X, _ in client_data]
    plt.bar(range(len(client_sizes)), client_sizes, color='skyblue')
    plt.title('Number of Samples per Client')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(client_sizes)))

    # Plot 2: Target distribution per client
    plt.subplot(2, 1, 2)
    for i, (_, client_y) in enumerate(client_data):
        if len(client_y) > 0:  # Only plot if client has data
            plt.hist(client_y, alpha=0.5, bins=20, label=f'Client {i}')

    plt.title('Temperature Distribution per Client')
    plt.xlabel('Maximum Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('data/processed/data_distribution.png')
    plt.close()

    print("Data distribution visualization saved to data/processed/data_distribution.png")


# Main function to run the entire data processing pipeline
def main():
    print("Starting NOAA weather data processing for federated learning...")

    # Step 1: Download weather data
    print("\n=== Downloading Weather Data ===")
    raw_data = download_weather_data()

    # Step 2: Process the data
    print("\n=== Processing Weather Data ===")
    processed_data = process_weather_data(raw_data)

    # Step 3: Create time series features
    print("\n=== Creating Time Series Features ===")
    window_size = 7  # Use past 7 days to predict
    forecast_horizon = 1  # Predict 1 day ahead
    target_col = 'TMAX'  # Predict maximum temperature
    X, y, stations = create_time_series_features(
        processed_data,
        window_size=window_size,
        target_col=target_col,
        forecast_horizon=forecast_horizon
    )

    # Step 4: Split data for federated learning
    print("\n=== Creating Non-IID Data Splits ===")
    num_clients = 5  # Number of simulated edge devices
    client_data = create_non_iid_splits(X, y, stations, num_clients=num_clients)

    # Step 5: Visualize the data distribution
    print("\n=== Visualizing Data Distribution ===")
    visualize_data_distribution(client_data)

    print("\nData processing completed! You can now proceed to build your LSTM model.")


if __name__ == "__main__":
    main()