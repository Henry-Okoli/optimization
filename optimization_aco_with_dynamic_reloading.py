
from collections import defaultdict
import random
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import os


# Global variables for ACO 
NUM_SIMULATIONS = 3  
NUM_ITERATIONS = 3
NUM_ANTS = 2         
PARTICLE_POPULATION = 5  
PHEROMONE_EVAPORATION_RATE = 0.5 
PHEROMONE_DEPOSIT_RATE = 1.0
INERTIA_WEIGHT = 0.7 
COGNITIVE_COEFFICIENT = 1.5 
SOCIAL_COEFFICIENT = 2.0

itype = '2W3DC'

# Load location data
locations_df = pd.read_csv(f"{itype}.csv")

# Load fleet data
fleet_df = pd.read_csv("fleet_Data.csv")

# Create an output folder if it doesn't exist
output_folder = f"output/{itype}"
os.makedirs(output_folder, exist_ok=True)

# Define function to calculate distance using haversine formula
def calculate_distance(coord1, coord2):
    return haversine(coord1, coord2, unit=Unit.KILOMETERS)

# Create a dictionary to store location coordinates
location_coords = {}

# Helper Function
def generate_cordinates(location_data):
    coords = {}
    for index, row in location_data.iterrows():
        coords[row['code']] = (row['latitude'], row['longitude'])
    return coords

# Calculate distance matrix
distance_matrix = np.zeros((len(locations_df), len(locations_df)))

def generate_distanceMatrix(location_data, coords):
    dmatrix = np.zeros((len(location_data), len(location_data)))
    for i in range(len(location_data)):
        for j in range(len(location_data)):
            dmatrix[i, j] = calculate_distance(coords[location_data['code'][i]], coords[location_data['code'][j]])
    return dmatrix

# Create a mapping dicitionary
location_index_mapping = {}
def generate_index_mapping(location_data):
    return {code: idx for idx, code in enumerate(location_data['code'])}

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

def aco(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, simulation_folder, cluster, locations_df, alpha=1, beta=5, evaporation_rate=0.5, deposit_rate=1.0, num_ants=50, iterations=100):
    best_route = None
    best_cost = float('inf')
    current_vehicle = None
    best_distance = float('inf')
    total_fuel_consumed = float('inf')

    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}

    # Determine distribution centers based on start location
    if start_location.startswith('M'):
        distribution_centers = set(locations_df[locations_df['code'].str.startswith('D')]['code'])
        distribution_centers.add(start_location)
    else:
        distribution_centers = set(locations_df[(locations_df['ClusterCode'] == cluster) & (locations_df['code'].str.startswith('D'))]['code'])

    for _ in range(iterations):
        print(f'    Iteration {_}')

        ants = [Ant(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, alpha, beta, location_index_mapping, locations_df, distribution_centers) for _ in range(num_ants)]

        for ant in ants:
            ant.construct_route()

            if ant.total_cost < best_cost:
                best_cost = ant.total_cost
                best_route = ant.route
                best_distance = ant.total_distance
                current_vehicle = ant.current_vehicle
                total_fuel_consumed = ant.total_fuel_consumed

        for ant in ants:
            ant.update_pheromone(pheromone_matrix, evaporation_rate, deposit_rate)

        # Save only the best ant's data for this iteration
        iteration_pd = pd.DataFrame([[best_cost, best_distance, total_fuel_consumed, best_route]], 
                                    columns=['total_cost', 'total_distance', 'total_fuel_consumed', 'route'])
        iteration_pd.to_csv(os.path.join(simulation_folder, f'ants_iteration{_}.csv'), index=False)

    return total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost

def choose_vehicle(start_location, end_locations, vehicles):
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

def run_simulations(start_location, end_locations, vehicles, distance_matrix, location_index_mapping, simulation_folder, cluster, locations_df, num_simulations=10):
    best_routes = {}
    best_costs = {}

    print(f"Running {num_simulations} ACO simulations for {cluster}")
    best_routes["ACO"] = []
    best_costs["ACO"] = []
    best_details = []

    for sim in range(num_simulations):
        print(f'Simulation {sim}')
        output_folder = os.path.join(simulation_folder, f'simulation_{sim}')
        os.makedirs(output_folder, exist_ok=True)
        
        pheromone_matrix = np.ones((len(locations_df), len(locations_df)))  # Set initial pheromone levels to 1
        
        total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost = aco(
            start_location, end_locations, vehicles, distance_matrix, pheromone_matrix,
            simulation_folder=output_folder, cluster=cluster, locations_df=locations_df,
            evaporation_rate=PHEROMONE_EVAPORATION_RATE, deposit_rate=PHEROMONE_DEPOSIT_RATE,
            num_ants=NUM_ANTS, iterations=NUM_ITERATIONS
        )
        
        best_routes["ACO"].append(best_route)
        best_costs["ACO"].append(best_cost)
        
        best_details.append([f'Simulation_{sim}', best_distance, best_cost, total_fuel_consumed, best_route])
     
        visualize_route(best_route, current_vehicle, distance_matrix, output_folder, cluster)
        save_route_data(best_route, current_vehicle, distance_matrix, output_folder, cluster, location_index_mapping, cycle_num=1)

    simulation_df = pd.DataFrame(best_details, columns=['Simulation', 'Distance Traveled', 'Cost Incurred', 'Fuel Consumed', 'Best Route'])
    simulation_df.to_csv(os.path.join(simulation_folder, f"simulation_data_{cluster}.csv"), index=False)

    return best_routes, best_costs

def visualize_route(route, vehicle, distance_matrix, simulation_folder, cluster):
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

def simulation(item_type, source, start_location, end_locations_cluster1):
    best_costs = 0
    if not end_locations_cluster1:
        return 0
    
    for cluster, end_locations in [(source, end_locations_cluster1)]:
        simulation_folder = os.path.join(output_folder,  item_type, start_location)
        os.makedirs(simulation_folder, exist_ok=True)

        if start_location.startswith('M') or start_location.startswith('W'):
            vehicles = fleet_df[fleet_df['Cluster'] == 'Core']
        else:
            vehicles = fleet_df[fleet_df['Cluster'] == cluster.replace(' ', '_')]

        best_routes, best_costs = run_simulations(
            start_location, end_locations, vehicles, distance_matrix,
            location_index_mapping, simulation_folder, cluster, locations_df,
            num_simulations=NUM_SIMULATIONS
        )

        best_algorithm = min(best_costs, key=best_costs.get)
        best_route = best_routes[best_algorithm][np.argmin(best_costs[best_algorithm])]

        visualize_route(best_route, choose_vehicle(start_location , end_locations, vehicles), distance_matrix, simulation_folder, cluster)
        save_route_data(best_route, choose_vehicle(start_location ,end_locations, vehicles), distance_matrix, simulation_folder, cluster, location_index_mapping, cycle_num=1)

        print(f"Best Algorithm for Cluster {cluster}: {best_algorithm}")
        print(f"Best Cost for Cluster {cluster}: {min(best_costs[best_algorithm])}")
        print()
        best_costs = min(best_costs[best_algorithm])
    
    return best_costs

def baseRun():
      
    items = []

    # Step 1:   We Get the Cost of Supplying the Warehouses from the Purchase Center.  We start from the Purchase Center (M) and then travel to all the Warehouses in a Specific Cluster and supply them 
    start_location = 'M1'  # Core (M1)
    warehouses_cluster1 = locations_df[(locations_df['ClusterCode'] == 'Cluster1') & (locations_df['code'].str.startswith('W'))]['code'].tolist()
    warehouses_cluster2 = locations_df[(locations_df['ClusterCode'] == 'Cluster2') & (locations_df['code'].str.startswith('W'))]['code'].tolist()
    #  'PC_Warehouse','Cluster1', 
    
    best_cost = simulation('PC_Warehouse','Cluster1', start_location,  warehouses_cluster1)
    items.append({'source': start_location , 'destination' : 'Warehouse' , 'Cluster': 'Cluster1' , 'cost': best_cost})
    
    best_cost = simulation('PC_Warehouse','Cluster2',start_location,  warehouses_cluster2)
    items.append({'source': start_location , 'destination' : 'Warehouse' , 'Cluster': 'Cluster2' , 'cost': best_cost})


    # Step 2: We get the Best Cost of Supplying all the Distribution Centers form the Warehouses in the Clusters

    
    distributions_cluster1 = locations_df[(locations_df['ClusterCode'] == 'Cluster1') & (locations_df['code'].str.startswith('D'))]['code'].tolist()
    distributions_cluster2 = locations_df[(locations_df['ClusterCode'] == 'Cluster2') & (locations_df['code'].str.startswith('D'))]['code'].tolist()
    
    for cluster, warehouse_locations , dist_locations in [('Cluster1', warehouses_cluster1 , distributions_cluster1), ('Cluster2', warehouses_cluster2 , distributions_cluster2)]:
        for Wh in warehouse_locations:
                start_location = Wh  # Go from the Warhouses
                best_cost = simulation('Warehouse_Distribution',cluster,start_location, dist_locations)
                items.append({'source': start_location , 'destination' : 'Distribution' , 'Cluster': cluster , 'cost': best_cost})

    # Step 3: We then get the Best Cost of Supplying all the RT from the Distribution Centers in the Cluster. 

    outlets_cluster1 = locations_df[(locations_df['ClusterCode'] == 'Cluster1') & (locations_df['code'].str.startswith('R'))]['code'].tolist()
    outlets_cluster2 = locations_df[(locations_df['ClusterCode'] == 'Cluster2') & (locations_df['code'].str.startswith('R'))]['code'].tolist()
    
    for cluster, dc_locations , outlet_locations in [('Cluster1', distributions_cluster1 , outlets_cluster1), ('Cluster2', distributions_cluster2 , outlets_cluster2)]:
        for dc in dc_locations:
                start_location = dc  # Go from the each of the DC
                best_cost = simulation('Distribution_RetailOutlet',cluster,start_location, outlet_locations)
                items.append({'source': start_location , 'destination' : 'Outlet' , 'Cluster': cluster , 'cost': best_cost})

    # Step 5: Let us output the best cost for the Clusters taking into consideration all the routes
    cluster_costs = defaultdict(lambda: {'Warehouse': np.inf, 'Distribution': np.inf, 'Outlet': np.inf})
    rows = []

    for item in items:
        cluster = item['Cluster']
        destination = item['destination']
        cost = item['cost']
        if cost < cluster_costs[cluster][destination]:
            cluster_costs[cluster][destination] = cost

            

    # Calculate the total cost for each cluster
    for cluster, values in cluster_costs.items():
        warehouse = values['Warehouse']
        distribution = values['Distribution']
        outlet = values['Outlet']
        total = warehouse + distribution + outlet
        rows.append([cluster, warehouse, distribution, outlet, total])


    print(rows)



        
    # Create DataFrame
    df = pd.DataFrame(rows, columns=['Cluster', 'Warehouse Cost', 'Distribution Cost', 'Outlet Cost', 'Total Cost'])

    # Save to CSV
    df.to_csv(f'{output_folder}/summary.csv', index=False)   

class Ant:
    def __init__(self, start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, alpha, beta, location_index_mapping, locations_df, distribution_centers):
        self.start_location = start_location
        self.current_location = start_location
        self.end_locations = set(end_locations)
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.route = [start_location]
        self.total_cost = 0
        self.total_distance = 0
        self.total_fuel_consumed = 0
        self.location_index_mapping = location_index_mapping
        self.locations_df = locations_df
        self.distribution_centers = distribution_centers
        self.location_capacities = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.precomputed_distances = defaultdict(dict)
        # Choose the appropriate vehicle based on the start location
        self.current_vehicle = choose_vehicle(start_location, end_locations, vehicles)
        self.current_load = self.current_vehicle['Capacity_KG']
        self.location_demands = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.unserviced_locations = set(end_locations)
        self.start_type = start_location[0]  # 'M', 'W', or 'D'
        self.end_type = end_locations[0][0] if end_locations else None  # 'W', 'D', or 'R'

    def construct_route(self, max_iterations=1000):
        iterations = 0
        while self.unserviced_locations and iterations < max_iterations:
            iterations += 1
            
            if iterations % 100 == 0:
                print(f"Iteration {iterations}: Current location {self.current_location}, Remaining locations: {self.unserviced_locations}")

            if self.start_type == 'M' and self.current_location == self.start_location:
                next_location = self.force_end_location_visit()
            else:
                next_location = self.select_next_location()

            if next_location is None:
                print(f"No valid next location found. Breaking loop.")
                break

            if next_location in self.distribution_centers:
                self.route.append(next_location)
                self.current_load = self.current_vehicle['Capacity_KG']
                print(f"Restocked at DC {next_location}. Current load: {self.current_load}")
            else:
                required_load = self.location_demands[next_location]
                
                if self.current_load >= required_load:
                    self.route.append(next_location)
                    self.current_load -= required_load
                    self.unserviced_locations.remove(next_location)
                    print(f"Serviced {next_location}. Remaining load: {self.current_load}")
                else:
                    nearest_dc = self.find_nearest_dc()
                    if nearest_dc == self.current_location:
                        print(f"Already at nearest DC {nearest_dc}. Breaking loop to avoid infinite restocking.")
                        break
                    self.route.append(nearest_dc)
                    self.current_load = self.current_vehicle['Capacity_KG']
                    print(f"Insufficient load. Restocked at nearest DC {nearest_dc}")
            
            self.update_costs(self.route[-2], self.route[-1])
            self.current_location = self.route[-1]

            # Check if we're stuck in a loop
            if len(self.route) > 3 and len(set(self.route[-3:])) == 2:
                print("Detected a potential loop. Forcing a jump to an unserviced location.")
                if self.unserviced_locations:
                    forced_location = random.choice(list(self.unserviced_locations))
                    self.route.append(forced_location)
                    self.current_location = forced_location
                    self.current_load = max(0, self.current_load - self.location_demands[forced_location])
                    self.unserviced_locations.remove(forced_location)

        if iterations == max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached. Route may be incomplete.")
        print(f"Route construction completed in {iterations} iterations.")
        print(f"Final route: {self.route}")
        print(f"Remaining unserviced locations: {self.unserviced_locations}")
    
    def select_next_location(self):
        probabilities = self.calculate_probabilities()
        if not probabilities:
            # If no valid probabilities, try to find any unserviced location or distribution center
            available_locations = list(self.unserviced_locations) + list(self.distribution_centers)
            if available_locations:
                return random.choice(available_locations)
            else:
                return None  # No valid location found
        return random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]

    def force_end_location_visit(self):
        if self.end_locations:
            return min(self.end_locations, key=lambda loc: self.get_distance(self.current_location, loc))
        return self.find_nearest_dc()

    
    def calculate_probabilities(self):
        probabilities = {}
        total_probability = 0
        
        for location in self.unserviced_locations | self.distribution_centers:
            if location == self.current_location:
                continue
            
            distance = self.get_distance(self.current_location, location)
            if distance == 0:
                continue
            
            pheromone = self.pheromone_matrix[self.location_index_mapping[self.current_location]][self.location_index_mapping[location]]
            probability = pheromone**self.alpha * (1/distance)**self.beta
            
            if location in self.unserviced_locations:
                required_load = self.location_demands[location]
                if self.current_load >= required_load:
                    probability *= 3.0  # Strongly favor serviceable locations
                else:
                    probability *= 0.5  # Reduce probability for locations we can't fully service
            elif location in self.distribution_centers:
                if self.current_load < 0.2 * self.current_vehicle['Capacity_KG']:
                    probability *= 2.0  # Favor distribution centers when load is low
                elif self.start_type in ['M', 'W'] and self.end_type in ['W', 'D']:
                    probability *= 1.5  # Slightly favor distribution centers for M->W and W->D routes
            
            probabilities[location] = probability
            total_probability += probability

        if total_probability == 0:
            return {}
        
        return {k: v / total_probability for k, v in probabilities.items()}

    def find_nearest_dc(self):
        return min(self.distribution_centers, key=lambda dc: self.get_distance(self.current_location, dc))

    def get_distance(self, start, end):
        if start not in self.precomputed_distances or end not in self.precomputed_distances[start]:
            distance = getDistance(start, end, self.distance_matrix, self.location_index_mapping)
            self.precomputed_distances[start][end] = distance
            self.precomputed_distances[end][start] = distance
        return self.precomputed_distances[start][end]

    def update_costs(self, start, end):
        distance = self.get_distance(start, end)
        self.total_distance += distance
        self.total_cost += calculate_cost([start, end], self.current_vehicle, self.distance_matrix, self.location_index_mapping)
        self.total_fuel_consumed += calculate_fuel_consumption(distance, self.current_vehicle, self.current_load)

    def update_pheromone(self, pheromone_matrix, evaporation_rate, deposit_rate):
        for i in range(len(self.route) - 1):
            start_location = self.route[i]
            end_location = self.route[i + 1]
            start_idx = self.location_index_mapping[start_location]
            end_idx = self.location_index_mapping[end_location]
            pheromone_matrix[start_idx][end_idx] *= (1 - evaporation_rate)
            pheromone_matrix[start_idx][end_idx] += deposit_rate / self.total_cost

if __name__ == "__main__": 
    df = locations_df
    
    location_coords = generate_cordinates(df)
    distance_matrix = generate_distanceMatrix(df,location_coords)
    location_index_mapping = generate_index_mapping(df)

    # Run the Base
    baseRun()

    # Separate the data into clusters
    clusters = df['ClusterCode'].unique()

    # Process each cluster
    for cluster in clusters:
        cluster_df = df[df['ClusterCode'] == cluster]
        
        # Extract W and D codes
        W_codes = cluster_df[cluster_df['code'].str.startswith('W')]['code'].tolist()
        D_codes = cluster_df[cluster_df['code'].str.startswith('D')]['code'].tolist()
        
        # Generate combinations of 1 W and 1 D to eliminate
        for w_code in W_codes:
            for d_code in D_codes:
                # Create a new dataframe excluding the selected W and D codes
                new_df = cluster_df[~cluster_df['code'].isin([w_code, d_code])]

                # Ensure M1 is included in the new dataframe
                if 'M1' not in new_df['code'].values:
                    m1_row = df[df['code'] == 'M1']
                    new_df = pd.concat([new_df, m1_row], ignore_index=True)
                
                # Save the new dataframe to a CSV file
                os.makedirs('input', exist_ok=True)
                itype = f'1W2DC - {w_code} and {d_code}'
                new_df.to_csv(f"input/{itype}.csv", index=False)
                

                output_folder = f"output/{cluster}/{itype}"
                os.makedirs(output_folder, exist_ok=True)

                
                # Simulate for each new Data Frame
                locations_df = pd.read_csv(f"input/{itype}.csv")
                location_coords = generate_cordinates(locations_df)
                distance_matrix = generate_distanceMatrix(locations_df,location_coords)
                location_index_mapping = generate_index_mapping(locations_df)

                baseRun()

        