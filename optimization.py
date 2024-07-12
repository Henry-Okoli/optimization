
import random
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import os


# Global variables for ACO and PSO parameters
NUM_SIMULATIONS = 5  
NUM_ITERATIONS = 5
NUM_ANTS = 5         
PARTICLE_POPULATION = 5  
PHEROMONE_EVAPORATION_RATE = 0.5 
PHEROMONE_DEPOSIT_RATE = 1.0
INERTIA_WEIGHT = 0.7 
COGNITIVE_COEFFICIENT = 1.5 
SOCIAL_COEFFICIENT = 2.0



# Define function to calculate distance using haversine formula
def calculate_distance(coord1, coord2):
    return haversine(coord1, coord2, unit=Unit.KILOMETERS)

# Load location data
locations_df = pd.read_csv("locations.csv")

# Load fleet data
fleet_df = pd.read_csv("fleet_Data.csv")

# Create a dictionary to store location coordinates
location_coords = {}
for index, row in locations_df.iterrows():
    location_coords[row['code']] = (row['latitude'], row['longitude'])

# Calculate distance matrix
distance_matrix = np.zeros((len(locations_df), len(locations_df)))
for i in range(len(locations_df)):
    for j in range(len(locations_df)):
        distance_matrix[i, j] = calculate_distance(location_coords[locations_df['code'][i]], location_coords[locations_df['code'][j]])
# print(distance_matrix)

# Create a mapping dicitionary
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
    # return (total_distance / total_cycle_distance) * 
    return (monthly_cost / 4)

def calculate_fuel_consumption(distance, vehicle, current_load):
    max_load = vehicle['Capacity_KG']
    fuel_max_load = vehicle['Fuel_Consumption_at_max_load_kmpg']
    fuel_zero_load = vehicle['Fuel_consumption_at_zero_load_kmpg']
    
    fuel_consumption = (
        (current_load / max_load) * fuel_max_load +
        (1 - current_load / max_load) * fuel_zero_load
    )
    return  distance / fuel_consumption


def calculate_cost(route, vehicle, distance_matrix, location_index_mapping):
    """Calculates the total cost of a route, including fuel, wear and tear, and maintenance."""
    total_cost = 0
    current_load = vehicle.Capacity_KG
    total_distance = 0
    # print(vehicle)

    for i in range(len(route) - 1):
        start_location = route[i]
        end_location = route[i + 1]
        distance = getDistance(start_location, end_location, distance_matrix, location_index_mapping) # distance_matrix[start_location, end_location]
        total_distance += distance

        # Calculate fuel consumption
        fuel_consumption = calculate_fuel_consumption(distance, vehicle, current_load)
        total_cost += fuel_consumption * 720 # -- This being the cost of fuel per  liter

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

def aco(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, simulation_folder, cluster ,alpha=1, beta=5, evaporation_rate=0.5, deposit_rate=1.0, num_ants=50, iterations=100):
    """Implements the Ant Colony Optimization algorithm."""
    best_route = None
    best_cost = float('inf')
    current_vehicle = None
    best_distance = float('inf')
    total_fuel_consumed = float('inf')

    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}
    

    for _ in range(iterations):
        # Log Each Iteration Info
        print(f'    Interation {_}')
        iteration_best_route = None
        iteration_best_cost = float('inf')
        iteration_end_locations = None
        iteration_distance_matrix = None
        iteration_current_vehicle =None

        # Create ant colony
        ants = [Ant(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, alpha, beta, location_index_mapping) for _ in range(num_ants)]

        # Let ants explore the graph
        for ant in ants:
            ant.construct_route()

        # Update pheromone levels
        for ant in ants:
            ant.update_pheromone(pheromone_matrix, evaporation_rate, deposit_rate)

        # Find the best route
        for ant in ants:
          #  print(f'Simulation {_} : Cost {ant.total_cost} ')
            if ant.total_cost < best_cost:
                best_cost = ant.total_cost
                best_route = ant.route
                best_distance = ant.total_distance
                current_vehicle = ant.current_vehicle
                total_fuel_consumed = ant.total_fuel_consumed

            # Best in Iteration
            if ant.total_cost < iteration_best_cost:
                iteration_best_cost = ant.total_cost
                iteration_best_route = ant.route
                iteration_end_locations = ant.end_locations
                iteration_distance_matrix = ant.distance_matrix
                iteration_current_vehicle = ant.current_vehicle


        
        
    
    # simulation.to_csv(os.path.join(simulation_folder, f'simulation_{_}.csv'), index=False)
    return total_fuel_consumed, current_vehicle ,best_distance, best_route, best_cost

def pso(start_location, end_locations, vehicles, distance_matrix,  inertia_weight=0.7, cognitive_coefficient=1.5, social_coefficient=2.0, particle_population=50, iterations=100):
    """Implements the Particle Swarm Optimization algorithm."""
    best_route = None
    best_cost = float('inf')

    
    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}


    # Initialize particle swarm
    particles = [Particle(start_location, end_locations, vehicles, distance_matrix, location_index_mapping) for _ in range(particle_population)]

    for _ in range(iterations):
        # Update particle velocities and positions
        for particle in particles:
            particle.update_velocity(inertia_weight, cognitive_coefficient, social_coefficient, particles)
            particle.update_position()

        # Find the best route
        for particle in particles:
            if particle.cost < best_cost:
                best_cost = particle.cost
                best_route = particle.route

    return best_route, best_cost


def greedy_algorithm(start_location, end_locations, vehicles, distance_matrix):
    """Implements the Greedy Algorithm for route optimization."""
    current_location = start_location
    route = [start_location]
    remaining_locations = end_locations.copy()
    current_vehicle = None
    current_load = 0

    while remaining_locations:
        # Select the closest location from remaining_locations
        closest_location = min(remaining_locations, key=lambda x: distance_matrix[current_location, x])

        # Check if vehicle capacity is sufficient
        closest_location_index = locations_df[locations_df['code'] == closest_location].index[0]
        required_capacity = locations_df['Capacity_KG'][closest_location_index]

        if current_vehicle is None or current_load + required_capacity <= current_vehicle.Capacity_KG:
            # Assign vehicle and update load
            if current_vehicle is None:
                # Select a suitable vehicle based on capacity and cluster
                current_vehicle = choose_vehicle(end_locations, vehicles)  # Assuming you have a function to choose vehicles
                current_load = 0

            # Update route and load
            route.append(closest_location)
            current_load += required_capacity
            remaining_locations.remove(closest_location)
            current_location = closest_location
        else:
            # Vehicle capacity reached, return to start and choose a new vehicle
            route.append(start_location)
            current_vehicle = None
            current_load = 0
            current_location = start_location

    return route


def choose_vehicle(end_locations, vehicles):
    """Chooses a suitable vehicle based on capacity and cluster."""
    # Assuming that vehicles are assigned to clusters and cannot be moved between clusters

    # Identify the cluster of the end locations
    cluster = locations_df[locations_df['code'].isin(end_locations)]['ClusterCode'].unique()[0]

    # Filter vehicles based on cluster
    cluster_vehicles = vehicles[vehicles['Cluster'] == cluster]

    # Choose a vehicle with sufficient capacity (assuming there is at least one)
    suitable_vehicles = cluster_vehicles[cluster_vehicles['Capacity_KG'] >= sum(locations_df[locations_df['code'].isin(end_locations)]['Capacity_KG'])]
    # print (cluster_vehicles)
    # Select a vehicle randomly from the suitable vehicles
    return cluster_vehicles.sample(1).iloc[0]

def run_simulations(start_location, end_locations, vehicles, distance_matrix, location_index_mapping, simulation_folder, cluster, num_simulations=10):
    """Runs multiple simulations for each optimization algorithm and returns the best route for each."""
    best_routes = {}
    best_costs = {}

    # Run ACO simulations
    
    print(f"Running ACO")
    best_routes["ACO"] = []
    best_costs["ACO"] = []
    best_details = []
    for _ in range(num_simulations):
        # Initialize pheromone matrix
        print(f'Simulation {_}')
        pheromone_matrix = np.ones((len(locations_df), len(locations_df)))  # Set initial pheromone levels to 1
        
        total_fuel_consumed, current_vehicle,best_distance, best_route, best_cost = aco(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix,simulation_folder=simulation_folder, cluster= cluster, evaporation_rate=PHEROMONE_EVAPORATION_RATE, deposit_rate=PHEROMONE_DEPOSIT_RATE, num_ants=NUM_ANTS, iterations=NUM_ITERATIONS)
        best_routes["ACO"].append(best_route)
        best_costs["ACO"].append(best_cost)
        # Save Data for the Simulation
        
        best_details.append( [f'Simulation_{_}' , best_distance, best_cost , total_fuel_consumed ,best_route ] )
        output_folder =  os.path.join(simulation_folder, f'simulation_{_}')
        os.makedirs(output_folder, exist_ok=True)     
        visualize_route(best_route, current_vehicle, distance_matrix, output_folder, cluster)
        save_route_data(best_route, current_vehicle, distance_matrix, output_folder, cluster, location_index_mapping, cycle_num=1)


    simulation_df = pd.DataFrame(best_details, columns=[ 'Simulation', 'Distance Traveled', 'Cost Incurred','Fuel Consumed', 'Best Route'])
    simulation_df.to_csv(os.path.join(simulation_folder, f"simulation_data_{cluster}.csv"), index=False)
  

    return best_routes, best_costs


def visualize_route(route, vehicle, distance_matrix, simulation_folder, cluster):
    """Visualizes the optimized route and saves it as a PNG image."""
    # Extract coordinates of locations in the route
    route_coords = [location_coords[location] for location in route]

    # Extract coordinates of all locations in the cluster
    cluster_locations = locations_df[locations_df['ClusterCode'] == cluster]
    cluster_coords = [(row['latitude'], row['longitude']) for _, row in cluster_locations.iterrows()]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the route
    ax.plot(*zip(*route_coords), color='blue', marker='o', linestyle='-', linewidth=2, label='Route')

    # Plot all locations in the cluster
    ax.scatter(*zip(*cluster_coords), color='red', marker='x', label='Locations')

    # Set plot title and labels
    ax.set_title(f"Optimized Route - Cluster {cluster} - Vehicle: {vehicle['Vehicle Type']}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add legend
    ax.legend()

    # Save the plot to the simulation folder
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

    item_cost = calculate_cost(route, vehicle, distance_matrix, location_index_mapping)

    for i in range(len(route) - 1):
        start_location = route[i]
        end_location = route[i + 1]
        distance = getDistance(start_location, end_location, distance_matrix, location_index_mapping) # distance_matrix[start_location, end_location]
        total_distance += distance

        # Calculate fuel consumption
        fuel_consumption = calculate_fuel_consumption(distance, vehicle, current_load)
        total_fuel_consumed += fuel_consumption
        total_cost += 720 * fuel_consumption

        # Calculate wear and tear
        wear_tear_cost = distance * vehicle.additional_WearAndTear_at_Load * (current_load / vehicle.Capacity_KG)
        total_cost += wear_tear_cost

        # Update load based on destination
        end_location_index = locations_df[locations_df['code'] == end_location].index[0]
        current_load -= locations_df['Capacity_KG'][end_location_index]

        route_data.append([cycle_num, vehicle['Vehicle Type'], start_location, end_location, distance, current_load, fuel_consumption, wear_tear_cost])

        # Calculate maintenance cost
        maintenance_cost = calculate_maintenance_cost(total_distance, vehicle)
   # total_cost += maintenance_cost
        total_maintenance_cost += maintenance_cost

    # Create a Pandas DataFrame from the route data
    route_df = pd.DataFrame(route_data, columns=['Cycle', 'Vehicle Type', 'Start Location', 'End Location', 'Distance', 'Load at End', 'Fuel Consumed', 'Wear and Tear Cost'])

    # Save the DataFrame to a CSV file
    route_df.to_csv(os.path.join(simulation_folder, f"route_data_{cluster}.csv"), index=False)

    # Print a summary of the route
    # print(f"Cluster {cluster}:")
    # print(f"Vehicle Type: {vehicle['Vehicle Type']}")
    # print(f"Total Distance: {total_distance:.2f} km")
    # print(f"Total Fuel Consumed: {total_fuel_consumed:.2f} kmpg")
    # print(f"Total Maintenance Cost: {total_maintenance_cost:.2f}")
    # print(f"Total Wear and Tear Cost: {wear_tear_cost:.2f}")
    # print(f"Total Cost of Fuel: { 720 * total_fuel_consumed :.2f}")
    # print(f"Total Cost of item: {item_cost :.2f}")
    # print(f"Total Cost of Trip: {total_cost + total_maintenance_cost  :.2f}")    
    # print(f"Route: {route}")
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
        self.total_distance = 0
        self.current_load = 0
        self.total_fuel_consumed = 0
        self.current_vehicle = None
        self.location_index_mapping = location_index_mapping

    def construct_route(self):
        """Constructs the ant's route based on pheromone levels and distance."""
        # Select a suitable vehicle based on capacity and cluster
        self.current_vehicle = choose_vehicle(self.end_locations, self.vehicles)  # Assuming you have a function to choose vehicles

        while self.end_locations:
            # Calculate probabilities for each remaining location
            probabilities = self.calculate_probabilities(self.current_location, self.end_locations)
            # Select the next location based on probabilities
            next_location = self.select_next_location(probabilities)

            # Update route, load, and cost
            self.route.append(next_location)
            self.current_load += locations_df['Capacity_KG'][locations_df[locations_df['code'] == next_location].index[0]]
            self.total_cost += calculate_cost([self.current_location, next_location], self.current_vehicle, self.distance_matrix, self.location_index_mapping)
            distance = getDistance(self.current_location, next_location, self.distance_matrix, self.location_index_mapping)
            self.total_distance += distance
            self.total_fuel_consumed +=  calculate_fuel_consumption(distance, self.current_vehicle, self.current_load)
            self.current_location = next_location
            self.end_locations.remove(next_location)

    def calculate_probabilities(self, current_location, remaining_locations):
        """Calculates the probabilities for each remaining location."""
        probabilities = np.zeros(len(remaining_locations))
        total_probability = 0
        
        current_location_idx = self.location_index_mapping[current_location]
        
        for i, location in enumerate(remaining_locations):
            location_idx = self.location_index_mapping[location]
            distance = getDistance(current_location, location, self.distance_matrix, self.location_index_mapping)  # self.distance_matrix[current_location_idx, location_idx]
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
        # print(self.route)
        for i in range(len(self.route) - 1):
            start_location = self.route[i]
            end_location = self.route[i + 1]
            start_idx = self.location_index_mapping[start_location]
            end_idx = self.location_index_mapping[end_location]
            pheromone_matrix[start_idx, end_idx] *= (1 - evaporation_rate)
            pheromone_matrix[start_idx, end_idx] += deposit_rate / self.total_cost


class Particle:
    def __init__(self, start_location, end_locations, vehicles, distance_matrix, location_index_mapping):
        self.start_location = start_location
        self.end_locations = end_locations.copy()
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.route =  route_to_indices(self.end_locations, location_index_mapping) #[start_location]
        self.cost = float('inf')
        self.velocity = np.zeros(len(self.end_locations), dtype=int)
        self.best_route =  route_to_indices(self.end_locations, location_index_mapping) #None
        self.best_cost = float('inf')
        self.current_load = 0
        self.current_vehicle = None
        self.location_index_mapping = location_index_mapping

    def update_velocity(self, inertia_weight, cognitive_coefficient, social_coefficient, particles):
        """Updates the particle's velocity based on the inertia weight, cognitive and social coefficients, and the best positions of the particle and swarm."""
        # Select a suitable vehicle based on capacity and cluster
        self.current_vehicle = choose_vehicle(self.end_locations, self.vehicles)  # Assuming you have a function to choose vehicles
        print(self.best_route)
        print(self.end_locations)
        # Update velocity
        for i, location in enumerate(self.end_locations):
            # Calculate cognitive component
            print(i)
            cognitive_component = cognitive_coefficient * random.random() * (self.best_route[i] - self.route[i])

            # Calculate social component
            best_particle = min(particles, key=lambda p: p.best_cost)
            social_component = social_coefficient * random.random() * (best_particle.best_route[i] - self.route[i])

            # Update velocity
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_component + social_component

    def update_position(self):
        """Updates the particle's position based on its velocity."""
        # Update route
        for i, location in enumerate(self.end_locations):
            if random.random() < 0.5 and self.velocity[i] != 0:  # Probabilistic update
                self.route[i] = location
                self.current_load += locations_df['Capacity_KG'][locations_df[locations_df['code'] == location].index[0]]

        # Calculate cost
        self.cost = calculate_cost(self.route, self.current_vehicle, self.distance_matrix, self.location_index_mapping)

        # Update best position and cost if necessary
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_route = self.route.copy()



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
        best_routes, best_costs = run_simulations(start_location, end_locations, vehicles, distance_matrix,location_index_mapping, simulation_folder, cluster= cluster, num_simulations = NUM_SIMULATIONS)

        # Find the best route across all algorithms
        best_algorithm = min(best_costs, key=best_costs.get)
        best_route = best_routes[best_algorithm][np.argmin(best_costs[best_algorithm])]

        # Visualize and save the best route
        visualize_route(best_route, choose_vehicle(end_locations, vehicles), distance_matrix, simulation_folder, cluster)
        save_route_data(best_route, choose_vehicle(end_locations, vehicles), distance_matrix, simulation_folder, cluster, location_index_mapping, cycle_num=1)

        # Print the best algorithm and its cost
        print(f"Best Algorithm for Cluster {cluster}: {best_algorithm}")
        print(f"Best Cost for Cluster {cluster}: {min(best_costs[best_algorithm])}")
        print()

    # Generate a summary of all simulations and their costs
    summary_df = pd.DataFrame({
        'Algorithm': ['ACO', 'PSO', 'Greedy'],
        'Cluster 1 Best Cost': [min(best_costs['ACO']), min(best_costs['PSO']), min(best_costs['Greedy'])],
        'Cluster 2 Best Cost': [min(best_costs['ACO']), min(best_costs['PSO']), min(best_costs['Greedy'])]
    })
    summary_df.to_csv(os.path.join(output_folder, 'summary.csv'), index=False)