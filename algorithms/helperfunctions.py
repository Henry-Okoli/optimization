

from haversine import haversine, Unit
import numpy as np


# Choose the right vehicle to use for the trip
def choose_vehicle(start_location, end_locations, vehicles, locations_df):
    """Chooses a suitable vehicle based on capacity, cluster, and start location."""
    total_capacity_needed = sum(locations_df[locations_df['code'].isin(end_locations)]['Capacity_KG'])
    
    if start_location.startswith('M') or start_location.startswith('W'):
        cluster_vehicles = vehicles[vehicles['Cluster'] == 'Core']
    else:
        # Identify the cluster of the end locations
        cluster = locations_df[locations_df['code'].isin(end_locations)]['ClusterCode'].unique()[0]
        cluster_vehicles = vehicles[vehicles['Cluster'] == cluster]

    # Choose a vehicle with sufficient capacity (assuming there is at least one)
    suitable_vehicles = cluster_vehicles[cluster_vehicles['Capacity_KG'] >= total_capacity_needed]
    
    if suitable_vehicles.empty:
        # If no vehicle has sufficient capacity, choose the largest available
        return cluster_vehicles.loc[cluster_vehicles['Capacity_KG'].idxmax()]
    else:
        # Select the smallest suitable vehicle to minimize costs
        return suitable_vehicles.loc[suitable_vehicles['Capacity_KG'].idxmin()]


# Calculate the distance from a start location to an end location
def getDistance(current_location, target_location, distance_matrix, location_index_mapping):
    current_location_idx = location_index_mapping[current_location]
    location_idx = location_index_mapping[target_location]
    return distance_matrix[current_location_idx, location_idx]


def generate_distanceMatrix(location_data, coords):
    dmatrix = np.zeros((len(location_data), len(location_data)))
    for i in range(len(location_data)):
        for j in range(len(location_data)):
            dmatrix[i, j] = calculate_distance(coords[location_data['code'][i]], coords[location_data['code'][j]])
    return dmatrix


# Define function to calculate distance using haversine formula
def calculate_distance(coord1, coord2):
    return haversine(coord1, coord2, unit=Unit.KILOMETERS)

# Generate the Cordinates using the lat long
def generate_cordinates(location_data):
    coords = {}
    for index, row in location_data.iterrows():
        coords[row['code']] = (row['latitude'], row['longitude'])
    return coords
