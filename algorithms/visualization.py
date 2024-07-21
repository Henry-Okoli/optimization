from .costfunctions import *
from .helperfunctions import getDistance
import pandas as pd
import os
import matplotlib.pyplot as plt


def visualize_route(route, vehicle, distance_matrix, simulation_folder, cluster,location_coords, locations_df):
    """Visualizes the optimized route and saves it as a PNG image."""
    # Extract coordinates of locations in the route
    route_coords = [location_coords[location] for location in route]

    # Extract coordinates of all locations in the cluster
    cluster_locations = locations_df[locations_df['ClusterCode'] == cluster]
    cluster_coords = [(row['latitude'], row['longitude']) for _, row in cluster_locations.iterrows()]
    cluster_codes = [row['code'] for _, row in cluster_locations.iterrows()]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the route
    ax.plot(*zip(*route_coords), color='blue', marker='o', linestyle='-', linewidth=2, label='Route')

    # Plot all locations in the cluster
    # ax.scatter(*zip(*cluster_coords), color='red', marker='x', label='Locations')
    # Plot all locations in the cluster with their codes
    for (lat, lon), code in zip(cluster_coords, cluster_codes):
        ax.scatter(lat, lon, color='red', marker='x')
        ax.annotate(code, (lat, lon), textcoords="offset points", xytext=(0,5), ha='center')

    # Set plot title and labels
    ax.set_title(f"Optimized Route - {simulation_folder} - Vehicle: {vehicle['Vehicle Type']}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add legend
    ax.legend()

    # Save the plot to the simulation folder
    plt.savefig(os.path.join(simulation_folder, f"route_visualization.png"))
    plt.close()

def save_route_data(route, vehicle, distance_matrix, simulation_folder, cluster, location_index_mapping, locations_df, cycle_num):
    """Saves the route data to a CSV file in the simulation folder."""
    route_data = []
    current_load = vehicle.Capacity_KG
    total_cost = 0
    total_distance = 0
    total_fuel_consumed = 0
    total_maintenance_cost = 0

    item_cost = calculate_cost(route, vehicle, distance_matrix, location_index_mapping, locations_df)

    for i in range(len(route) - 1):
        start_location = route[i]
        end_location = route[i+1]
        distance = getDistance(start_location, end_location, distance_matrix, location_index_mapping)
        total_distance += distance

        # Calculate fuel consumption
        fuel_consumption = calculate_fuel_consumption(distance, vehicle, current_load)
        total_fuel_consumed += fuel_consumption
        fuel_cost = 720 * fuel_consumption
        total_cost += fuel_cost

        # Calculate wear and tear
        wear_tear_cost = distance * vehicle.additional_WearAndTear_at_Load * (current_load / vehicle.Capacity_KG)
        total_cost += wear_tear_cost

        # Calculate load change
        end_location_index = locations_df[locations_df['code'] == end_location].index[0]
        location_capacity = locations_df['Capacity_KG'][end_location_index]

        # Determine if location is for discharge or restock
        if end_location.startswith('D') or end_location.startswith('W'):
            # This is a distribution center or warehouse, so we restock
            discharged_restocked = vehicle.Capacity_KG - current_load
            new_load = vehicle.Capacity_KG
        else:
            # This is a retail outlet, so we discharge
            discharged_restocked = -min(current_load, location_capacity)
            new_load = max(0, current_load - location_capacity)

        route_data.append([
            cycle_num, 
            vehicle['Vehicle Type'], 
            start_location, 
            end_location, 
            distance, 
            current_load,  # Load at Start
            new_load,  # Load at End
            discharged_restocked,  # Discharged/Restocked
            fuel_consumption, 
            wear_tear_cost
        ])

        current_load = new_load

        # Calculate maintenance cost
        maintenance_cost = calculate_maintenance_cost(total_distance, vehicle)
        total_maintenance_cost += maintenance_cost

    # Create a Pandas DataFrame from the route data
    route_df = pd.DataFrame(route_data, columns=[
        'Cycle', 'Vehicle Type', 'Start Location', 'End Location', 'Distance', 
        'Load at Start', 'Load at End', 'Discharged/Restocked', 
        'Fuel Consumed', 'Wear and Tear Cost'
    ])

    # Save the DataFrame to a CSV file
    route_df.to_csv(os.path.join(simulation_folder, f"route_data_{cluster}.csv"), index=False)

    print(f"Route data saved to {os.path.join(simulation_folder, f'route_data_{cluster}.csv')}")

