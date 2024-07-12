import random
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import os

# Global variables for ACO and PSO parameters
NUM_SIMULATIONS = 100
NUM_ANTS = 10
PARTICLE_POPULATION = 10
PHEROMONE_EVAPORATION_RATE = 0.5
PHEROMONE_DEPOSIT_RATE = 1.0
INERTIA_WEIGHT = 0.7
COGNITIVE_COEFFICIENT = 1.5
SOCIAL_COEFFICIENT = 2.0

# Define function to calculate distance using haversine formula
def calculate_distance(coord1, coord2):
    return haversine(coord1, coord2, unit=Unit.KILOMETERS)

# Load location data
locations_df = pd.read_csv("optimization/locations.csv")

# Load fleet data
fleet_df = pd.read_csv("optimization/fleet_Data.csv")

# Create a dictionary to store location coordinates
location_coords = {}
for index, row in locations_df.iterrows():
    location_coords[row['code']] = (row['latitude'], row['longitude'])

# Calculate distance matrix
distance_matrix = np.zeros((len(locations_df), len(locations_df)))
for i in range(len(locations_df)):
    for j in range(len(locations_df)):
        distance_matrix[i, j] = calculate_distance(location_coords[locations_df['code'][i]], location_coords[locations_df['code'][j]])

# Create a mapping dictionary
location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}

# Create an output folder if it doesn't exist
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

def route_to_indices(route, location_index_mapping):
    return [location_index_mapping[loc] for loc in route]

def indices_to_route(indices, index_location_mapping):
    return [index_location_mapping[idx] for idx in indices]

def getDistance(current_location, target_location, distance_matrix, location_index_mapping):
    current_location_idx = location_index_mapping[current_location]
    location_idx = location_index_mapping[target_location]
    return distance_matrix[current_location_idx, location_idx]

# Maintenance cost calculation
def calculate_maintenance_cost(total_distance, vehicle):
    monthly_cost = vehicle['monthly_maintenance_cost']
    return (monthly_cost / 4)

def calculate_fuel_consumption(distance, vehicle, current_load):
    max_load = vehicle['Capacity_KG']
    fuel_max_load = vehicle['Fuel_Consumption_at_max_load_kmpg']
    fuel_zero_load = vehicle['Fuel_consumption_at_zero_load_kmpg']
    
    fuel_consumption = (
        (current_load / max_load) * fuel_max_load +
        (1 - current_load / max_load) * fuel_zero_load
    )
    return fuel_consumption * distance

def calculate_cost(route, vehicle, distance_matrix, location_index_mapping):
    """Calculates the total cost of a route, including fuel, wear and tear, and maintenance."""
    total_cost = 0
    current_load = vehicle.Capacity_KG
    total_distance = 0

    for i in range(len(route) - 1):
        start_location = route[i]
        end_location = route[i + 1]
        distance = getDistance(start_location, end_location, distance_matrix, location_index_mapping)
        total_distance += distance

        # Calculate fuel consumption
        fuel_consumption = calculate_fuel_consumption(distance, vehicle, current_load)
        total_cost += fuel_consumption

        # Calculate wear and tear
        wear_tear_cost = distance * vehicle.additional_WearAndTear_at_Load * (current_load / vehicle.Capacity_KG)
        total_cost += wear_tear_cost

        # Update load based on destination
        end_location_index = locations_df[locations_df['code'] == end_location].index[0]
        current_load -= locations_df['Capacity_KG'][end_location_index]

    # Calculate maintenance cost
    maintenance_cost = calculate_maintenance_cost(total_distance, vehicle)
    total_cost += maintenance_cost

    return total_cost

def aco(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, alpha=1, beta=5, evaporation_rate=0.5, deposit_rate=1.0, num_ants=50, iterations=100):
    """Implements the Ant Colony Optimization algorithm."""

    epsilon = 1e-10  # Small value to avoid division by zero
    distance_matrix_with_epsilon = distance_matrix + epsilon

    best_route = None
    best_cost = float('inf')

    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}

    for _ in range(iterations):
        # Create ant colony
        ants = [Ant(start_location, end_locations, vehicles, distance_matrix_with_epsilon, pheromone_matrix, alpha, beta, location_index_mapping) for _ in range(num_ants)]

        # Let ants explore the graph
        for ant in ants:
            ant.construct_route()

        # Update pheromone levels
        for ant in ants:
            ant.update_pheromone(pheromone_matrix, evaporation_rate, deposit_rate)

        # Find the best route
        for ant in ants:
            if ant.total_cost < best_cost:
                best_cost = ant.total_cost
                best_route = ant.route

    return best_route, best_cost

# Function definitions for PSO and Greedy Algorithm can be included here

def choose_vehicle(end_locations, vehicles):
    """Chooses a suitable vehicle based on capacity and cluster."""
    cluster = locations_df[locations_df['code'].isin(end_locations)]['ClusterCode'].unique()[0]
    cluster_vehicles = vehicles[vehicles['Cluster'] == cluster]
    suitable_vehicles = cluster_vehicles[cluster_vehicles['Capacity_KG'] >= sum(locations_df[locations_df['code'].isin(end_locations)]['Capacity_KG'])]
    return cluster_vehicles.sample(1).iloc[0]

def run_simulations(start_location, end_locations, vehicles, distance_matrix, location_index_mapping, num_simulations=10):
    """Runs multiple simulations for each optimization algorithm and returns the best route for each."""
    best_routes = {}
    best_costs = {}

    # Run ACO simulations
    print(f"Running ACO")
    best_routes["ACO"] = []
    best_costs["ACO"] = []
    for _ in range(num_simulations):
        pheromone_matrix = np.ones((len(locations_df), len(locations_df)))  # Set initial pheromone levels to 1
        best_route, best_cost = aco(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, evaporation_rate=PHEROMONE_EVAPORATION_RATE, deposit_rate=PHEROMONE_DEPOSIT_RATE, num_ants=NUM_ANTS)
        best_routes["ACO"].append(best_route)
        best_costs["ACO"].append(best_cost)

    # PSO and Greedy simulations can be added similarly

    return best_routes, best_costs

def visualize_route(route, vehicle, distance_matrix, simulation_folder, cluster):
    """Visualizes the optimized route and saves it as a PNG image."""
    route_coords = [location_coords[location] for location in route]
    cluster_locations = locations_df[locations_df['ClusterCode'] == cluster]
    cluster_coords = [(row['latitude'], row['longitude']) for _, row in cluster_locations.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(*zip(*route_coords), color='blue', marker='o', linestyle='-', linewidth=2, label='Route')
    ax.scatter(*zip(*cluster_coords), color='red', marker='x', label='Locations')

    ax.set_title(f"Optimized Route - Cluster {cluster} - Vehicle: {vehicle['Vehicle Type']}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    plt.savefig(os.path.join(simulation_folder, f"route_visualization.png"))
    plt.close()

def save_route_data(route, vehicle, distance_matrix, simulation_folder, cluster, location_index_mapping, cycle_num):
    """Saves the route data to a CSV file in the simulation folder."""
    route_data = []
    current_load = vehicle.Capacity_KG
    total_cost = 0
    total_distance = 0
    total_fuel_consumed = 0
    total_maintenance_cost = 0

    for i in range(len(route) - 1):
        start_location = route[i]
        end_location = route[i + 1]
        distance = getDistance(start_location, end_location, distance_matrix, location_index_mapping)
        total_distance += distance

        fuel_consumption = calculate_fuel_consumption(distance, vehicle, current_load)
        total_fuel_consumed += fuel_consumption

        wear_tear_cost = distance * vehicle.additional_WearAndTear_at_Load * (current_load / vehicle.Capacity_KG)
        total_cost += wear_tear_cost

        end_location_index = locations_df[locations_df['code'] == end_location].index[0]
        current_load -= locations_df['Capacity_KG'][end_location_index]

        route_data.append([cycle_num, vehicle['Vehicle Type'], start_location, end_location, distance, current_load, fuel_consumption, wear_tear_cost])

    maintenance_cost = calculate_maintenance_cost(total_distance, vehicle)
    total_cost += maintenance_cost
    total_maintenance_cost += maintenance_cost

  # Create a Pandas DataFrame from the route data
    route_df = pd.DataFrame(route_data, columns=['Cycle', 'Vehicle Type', 'Start Location', 'End Location', 'Distance', 'Load at End', 'Fuel Consumed', 'Wear and Tear Cost'])

    # Save the DataFrame to a CSV file
    route_df.to_csv(os.path.join(simulation_folder, f"route_data_{cluster}.csv"), index=False)

    print(f"Cluster {cluster}:")
    print(f"Vehicle Type: {vehicle['Vehicle Type']}")
    print(f"Total Distance: {total_distance:.2f} km")
    print(f"Total Fuel Consumed: {total_fuel_consumed:.2f} kmpg")
    print(f"Total Maintenance Cost: {total_maintenance_cost:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Route: {route}")
    print()

class Ant:
    def __init__(self, start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, alpha, beta, location_index_mapping):
        self.current_location = start_location
        self.start_location = start_location
        self.end_locations = end_locations.copy()
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.route = [start_location]
        self.total_cost = 0
        self.current_load = 0
        self.current_vehicle = None
        self.location_index_mapping = location_index_mapping

    def construct_route(self):
        """Constructs the ant's route based on pheromone levels and distance."""
        self.current_vehicle = choose_vehicle(self.end_locations, self.vehicles)

        while self.end_locations:
            probabilities = self.calculate_probabilities(self.current_location, self.end_locations)
            next_location = self.select_next_location(probabilities)

            self.route.append(next_location)
            self.current_load += locations_df['Capacity_KG'][locations_df[locations_df['code'] == next_location].index[0]]
            self.total_cost += calculate_cost([self.current_location, next_location], self.current_vehicle, self.distance_matrix, self.location_index_mapping)
            self.current_location = next_location
            self.end_locations.remove(next_location)

    def calculate_probabilities(self, current_location, remaining_locations):
        """Calculates the probabilities for each remaining location."""
        probabilities = np.zeros(len(remaining_locations))
        total_probability = 0
        
        current_location_idx = self.location_index_mapping[current_location]
        
        for i, location in enumerate(remaining_locations):
            location_idx = self.location_index_mapping[location]
            distance = getDistance(current_location, location, self.distance_matrix, self.location_index_mapping)
            pheromone = self.pheromone_matrix[current_location_idx, location_idx]

            probability = pheromone**self.alpha * (1/distance)**self.beta
            probabilities[i] = probability
            total_probability += probability

        return probabilities / total_probability

    def select_next_location(self, probabilities):
        """Selects the next location based on the calculated probabilities."""
        return random.choices(self.end_locations, weights=probabilities)[0]

    def update_pheromone(self, pheromone_matrix, evaporation_rate, deposit_rate):
        """Updates pheromone levels based on the ant's route."""
        for i in range(len(self.route) - 1):
            start_location = self.route[i]
            end_location = self.route[i + 1]
            start_idx = self.location_index_mapping[start_location]
            end_idx = self.location_index_mapping[end_location]
            pheromone_matrix[start_idx, end_idx] *= (1 - evaporation_rate)
            pheromone_matrix[start_idx, end_idx] += deposit_rate / self.total_cost

if __name__ == "__main__":
    # Define the start and end locations
    start_location = 'M1'  # Core (M1)
    end_locations_cluster1 = locations_df[locations_df['ClusterCode'] == 'Cluster1']['code'].tolist()
    end_locations_cluster2 = locations_df[locations_df['ClusterCode'] == 'Cluster2']['code'].tolist()

    # Run simulations for each cluster
    for cluster, end_locations in [('Cluster1', end_locations_cluster1), ('Cluster2', end_locations_cluster2)]:
        # Create a simulation folder
        simulation_folder = os.path.join(output_folder, cluster)
        os.makedirs(simulation_folder, exist_ok=True)

        # Filter vehicles based on cluster
        vehicles = fleet_df[fleet_df['Cluster'] == cluster.replace(' ', '_')]

        # Run simulations and get the best routes
        best_routes, best_costs = run_simulations(start_location, end_locations, vehicles, distance_matrix, location_index_mapping)

        # Find the best route across all algorithms
        best_algorithm = min(best_costs, key=lambda k: min(best_costs[k]))
        best_route = best_routes[best_algorithm][np.argmin(best_costs[best_algorithm])]

        # Visualize and save the best route
        visualize_route(best_route, choose_vehicle(end_locations, vehicles), distance_matrix, simulation_folder, cluster)
        save_route_data(best_route, choose_vehicle(end_locations, vehicles), distance_matrix, simulation_folder, cluster, location_index_mapping, cycle_num=1)

        # Print the best algorithm and its cost
        print(f"Best Algorithm for Cluster {cluster}: {best_algorithm}")
        print(f"Best Cost for Cluster {cluster}: {min(best_costs[best_algorithm])}")
        print()

    # Generate a summary of all simulations and their costs
    summary_data = {
        'Algorithm': [],
        'Cluster': [],
        'Best Cost': []
    }

    for cluster in ['Cluster1', 'Cluster2']:
        for algo in best_costs.keys():
            summary_data['Algorithm'].append(algo)
            summary_data['Cluster'].append(cluster)
            summary_data['Best Cost'].append(min(best_costs[algo]))

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_folder, 'summary.csv'), index=False)