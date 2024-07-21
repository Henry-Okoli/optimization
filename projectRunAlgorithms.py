
from collections import defaultdict
import pandas as pd
import numpy as np
import os
from algorithms.helperfunctions import  choose_vehicle,generate_cordinates,generate_distanceMatrix
from algorithms.visualization import visualize_route, save_route_data
from algorithms.ant import aco
from algorithms.pso import pso
from algorithms.greedy import greedy


# Global variables for ACO 
NUM_SIMULATIONS = 3  




# Global variables for PSO 


# My Variables
itype = '2W3DC' # Segment type to start with
locations_df = pd.read_csv(f"{itype}.csv") # Load location data from file
fleet_df = pd.read_csv("fleet_Data.csv")  # Load fleet data

output_folder = ''  # Create an output folder if it doesn't exist

location_coords = {} # Create a dictionary to store location coordinates


# Calculate distance matrix
distance_matrix = np.zeros((len(locations_df), len(locations_df)))


# Create a mapping dicitionary
location_index_mapping = {}
def generate_index_mapping(location_data):
    return {code: idx for idx, code in enumerate(location_data['code'])}

def route_to_indices(route, location_index_mapping):
    return [location_index_mapping[loc] for loc in route]

def indices_to_route(indices, index_location_mapping):
    return [index_location_mapping[idx] for idx in indices]

def run_simulations(algorithm , start_location, end_locations, vehicles, distance_matrix, location_index_mapping, simulation_folder, cluster, locations_df, num_simulations=10):
    best_routes = {'best_routes':['a','b']}
    best_costs = {'best_costs':[100,200,300]}

    print(f"Running {num_simulations} {algorithm} simulations for {cluster}")
    best_routes[algorithm] = []
    best_costs[algorithm] = []
    best_details = []

    for sim in range(num_simulations):
        print(f'Simulation {sim}')
        output_folder = os.path.join(simulation_folder, f'simulation_{sim}')
        os.makedirs(output_folder, exist_ok=True)
        
 
        if (algorithm == 'aco'):
            pheromone_matrix = np.ones((len(locations_df), len(locations_df)))  # Set initial pheromone levels to 1
            total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost = aco(
                start_location, end_locations, vehicles, distance_matrix, pheromone_matrix,
                simulation_folder=output_folder, cluster=cluster, locations_df=locations_df
            )

        if(algorithm== 'greedy'):
            total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost = greedy(
            start_location, end_locations, vehicles, distance_matrix, 
            simulation_folder=output_folder, cluster=cluster, locations_df=locations_df
        )
        
        if(algorithm== 'pso'):
            total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost = pso(
            start_location, end_locations, vehicles, distance_matrix, 
            simulation_folder=output_folder, cluster=cluster, locations_df=locations_df
        )
            
        best_routes[algorithm].append(best_route)
        best_costs[algorithm].append(best_cost)
        
        best_details.append([f'Simulation_{sim}', best_distance, best_cost, total_fuel_consumed, best_route])
     
        visualize_route(best_route, current_vehicle, distance_matrix, output_folder, cluster,location_coords, locations_df)
        save_route_data(best_route, current_vehicle, distance_matrix, output_folder, cluster, location_index_mapping,locations_df, cycle_num=1)

    simulation_df = pd.DataFrame(best_details, columns=['Simulation', 'Distance Traveled', 'Cost Incurred', 'Fuel Consumed', 'Best Route'])
    simulation_df.to_csv(os.path.join(simulation_folder, f"simulation_data_{cluster}.csv"), index=False)

    return best_routes, best_costs

def simulation(algorithm, item_type, source, start_location, end_locations_cluster1):
    best_cost = 0
    if not end_locations_cluster1:
        return 0
    
    for cluster, end_locations in [(source, end_locations_cluster1)]:
        simulation_folder = os.path.join( "output" ,algorithm ,cluster ,itype,  item_type, start_location)
        os.makedirs(simulation_folder, exist_ok=True)

        if start_location.startswith('M') or start_location.startswith('W'):
            vehicles = fleet_df[fleet_df['Cluster'] == 'Core']
        else:
            vehicles = fleet_df[fleet_df['Cluster'] == cluster.replace(' ', '_')]

        best_routes, best_costs = run_simulations(algorithm,
            start_location, end_locations, vehicles, distance_matrix,
            location_index_mapping, simulation_folder, cluster, locations_df,
            num_simulations=NUM_SIMULATIONS
        )
        
        
        print(best_costs)
        print(best_routes)

        best_route = best_routes[algorithm][np.argmin(best_costs[algorithm])]

        visualize_route(best_route, choose_vehicle(start_location , end_locations, vehicles, locations_df), distance_matrix, simulation_folder, cluster,location_coords, locations_df)
        save_route_data(best_route, choose_vehicle(start_location ,end_locations, vehicles, locations_df), distance_matrix, simulation_folder, cluster, location_index_mapping,  locations_df, cycle_num=1)

        print(f'Algorithm -- {algorithm}')
        print(f"Best Cost for Cluster {cluster}: {min(best_costs[algorithm])}")
        print()
        best_cost = min(best_costs[algorithm])
    
    return best_cost

def baseRun():
      
    
    listofalgorithms = ['pso'] # ,'aco','greedy'] 

    for algorithm in listofalgorithms:
        items = []
        clusterList = [cluster for cluster in locations_df['ClusterCode'].unique() if cluster != 'Core']

        for icluster in clusterList:
            output_folder = f"output/{algorithm}/{icluster}/{itype}"
            os.makedirs(output_folder, exist_ok=True)

            # Step 1:   We Get the Cost of Supplying the Warehouses from the Purchase Center.  We start from the Purchase Center (M) and then travel to all the Warehouses in a Specific Cluster and supply them 
            start_location = 'M1'  # Core (M1)
            warehouses_cluster = locations_df[(locations_df['ClusterCode'] == icluster) & (locations_df['code'].str.startswith('W'))]['code'].tolist()
            
            best_cost = simulation(algorithm,'PC_Warehouse',icluster, start_location,  warehouses_cluster)
            items.append({'algorithm':algorithm , 'source': start_location , 'destination' : 'Warehouse' , 'Cluster': icluster , 'cost': best_cost})
              
            # Step 2: We get the Best Cost of Supplying all the Distribution Centers form the Warehouses in the Clusters
 
            distributions_cluster = locations_df[(locations_df['ClusterCode'] == icluster) & (locations_df['code'].str.startswith('D'))]['code'].tolist()
       
            for Wh in warehouses_cluster:
                start_location = Wh  # Go from the Warhouses
                best_cost = simulation(algorithm,'Warehouse_Distribution',icluster,start_location, distributions_cluster)
                items.append({'algorithm':algorithm , 'source': start_location , 'destination' : 'Distribution' , 'Cluster': icluster , 'cost': best_cost})

            # Step 3: We then get the Best Cost of Supplying all the RT from the Distribution Centers in the Cluster. 

            outlets_cluster = locations_df[(locations_df['ClusterCode'] == icluster) & (locations_df['code'].str.startswith('R'))]['code'].tolist()
               
            for dc in distributions_cluster:
                    start_location = dc  # Go from the each of the DC
                    best_cost =  simulation(algorithm, 'Distribution_RetailOutlet',icluster,start_location, outlets_cluster)
                    items.append({'algorithm':algorithm , 'source': start_location , 'destination' : 'Outlet' , 'Cluster': icluster , 'cost': best_cost})

            # Step 5: Let us output the best cost for the Clusters taking into consideration all the routes
         
                
            # Create DataFrame
            df = pd.DataFrame(items, columns=['algorithm', 'source', 'destination','Cluster', 'Cost'])

            # Save to CSV
            df.to_csv(f'{output_folder}/summary.csv', index=False)   


if __name__ == "__main__": 
    df = locations_df   

    clusters = [cluster for cluster in df['ClusterCode'].unique() if cluster != 'Core']

    for cluster in clusters:
        itype = '2W3DC'
        locations_df = pd.read_csv(f"input/{itype}_{cluster}.csv")
        location_coords = generate_cordinates(locations_df)
        distance_matrix = generate_distanceMatrix(locations_df,location_coords)
        location_index_mapping = generate_index_mapping(locations_df)
        baseRun()

    segments = ['1W2DC']
    for segment in segments:
        # Separate the data into clusters
        for cluster in clusters:
            cluster_df = df[df['ClusterCode'] == cluster]

            # Extract W and D codes
            W_codes = cluster_df[cluster_df['code'].str.startswith('W')]['code'].tolist()
            D_codes = cluster_df[cluster_df['code'].str.startswith('D')]['code'].tolist()
            
            # Generate combinations of 1 W and 1 D to eliminate
            for w_code in W_codes:
                for d_code in D_codes:           
                    # Simulate for each new Data Frame
                    itype = f'{segment} - {w_code} and {d_code}'
                    locations_df = pd.read_csv(f"input/{itype}.csv")
                    location_coords = generate_cordinates(locations_df)
                    distance_matrix = generate_distanceMatrix(locations_df,location_coords)
                    location_index_mapping = generate_index_mapping(locations_df)
                    print(itype)
                    baseRun()

        