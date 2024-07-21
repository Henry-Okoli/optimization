
import os
import pandas as pd
import random
from collections import defaultdict
from .helperfunctions  import choose_vehicle , getDistance
from .costfunctions import calculate_cost, calculate_fuel_consumption



def greedy(start_location, end_locations, vehicles, distance_matrix, simulation_folder, cluster, locations_df):
    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}
    distribution_centers = set(locations_df[locations_df['code'].str.startswith('D')]['code'])
    distribution_centers.add(start_location)

    greedy_solver = GreedySolver(start_location, end_locations, vehicles, distance_matrix, location_index_mapping, locations_df, distribution_centers)
    
    best_route, best_cost = greedy_solver.solve()

    # Save data for this iteration
    iteration_pd = pd.DataFrame([[best_cost, best_route]], 
                                columns=['total_cost', 'route'])
    iteration_pd.to_csv(os.path.join(simulation_folder, f'greedy_result.csv'), index=False)

    current_vehicle = choose_vehicle(start_location, end_locations, vehicles)
    best_distance = sum(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping) for i in range(len(best_route)-1))
    total_fuel_consumed = sum(calculate_fuel_consumption(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping), current_vehicle, current_vehicle['Capacity_KG']) for i in range(len(best_route)-1))

    return total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost

class GreedySolver:
    def __init__(self, start_location, end_locations, vehicles, distance_matrix, location_index_mapping, locations_df, distribution_centers):
        self.start_location = start_location
        self.end_locations = list(end_locations)
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.location_index_mapping = location_index_mapping
        self.locations_df = locations_df
        self.distribution_centers = distribution_centers
        self.location_capacities = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.current_vehicle = choose_vehicle(start_location, end_locations, vehicles)
        self.current_load = self.current_vehicle['Capacity_KG']
        self.location_demands = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.unserviced_locations = set(end_locations)
        self.start_type = start_location[0]
        self.end_type = end_locations[0][0] if end_locations else None

    def solve(self):
        route = [self.start_location]
        current_location = self.start_location
        remaining_locations = self.end_locations.copy()

        while remaining_locations:
            next_location = self.find_nearest_location(current_location, remaining_locations)
            route.append(next_location)
            current_location = next_location
            remaining_locations.remove(next_location)

        cost = self.calculate_route_cost(route)
        return route, cost

    def find_nearest_location(self, current_location, locations):
        return min(locations, key=lambda x: self.get_distance(current_location, x))

    def get_distance(self, loc1, loc2):
        return getDistance(loc1, loc2, self.distance_matrix, self.location_index_mapping)

    def calculate_route_cost(self, route):
        return calculate_cost(route, self.current_vehicle, self.distance_matrix, self.location_index_mapping)

