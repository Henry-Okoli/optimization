
import os
import pandas as pd
import random
from collections import defaultdict
from .helperfunctions  import choose_vehicle , getDistance
from .costfunctions import calculate_cost, calculate_fuel_consumption


#default 

NUM_ANTS = 2         
PHEROMONE_EVAPORATION_RATE = 0.5 
PHEROMONE_DEPOSIT_RATE = 1.0
NUM_ITERATIONS = 3
ALPHA = 1
BETA = 5, 

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
        self.current_vehicle = choose_vehicle(start_location, end_locations, vehicles, locations_df)
        self.current_load = self.current_vehicle['Capacity_KG']
        self.location_demands = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.unserviced_locations = set(end_locations)
        self.start_type = start_location[0]  # 'M', 'W', or 'D'
        self.end_type = end_locations[0][0] if end_locations else None  # 'W', 'D', or 'R'

    def construct_route(self, max_iterations=100):
        iterations = 0

        while self.unserviced_locations and iterations < max_iterations:
            iterations += 1
            
            if iterations % 100 == 0:
                print(f"Iteration {iterations}: Current location {self.current_location}, Remaining locations: {self.unserviced_locations}")

        
            if self.start_type == 'M' and self.current_location == self.start_location:
                next_location = self.select_next_location()  # self.force_end_location_visit()
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
        print(self.distribution_centers)
        print('Here')
        return min((dc for dc in self.distribution_centers if dc != self.current_location), 
                   key=lambda dc: self.get_distance(self.current_location, dc))

    def get_distance(self, start, end):
        if start not in self.precomputed_distances or end not in self.precomputed_distances[start]:
            distance = getDistance(start, end, self.distance_matrix, self.location_index_mapping)
            self.precomputed_distances[start][end] = distance
            self.precomputed_distances[end][start] = distance
        return self.precomputed_distances[start][end]

    def update_costs(self, start, end):
        distance = self.get_distance(start, end)
        self.total_distance += distance
        self.total_cost += calculate_cost([start, end], self.current_vehicle, self.distance_matrix, self.location_index_mapping, self.locations_df)
        self.total_fuel_consumed += calculate_fuel_consumption(distance, self.current_vehicle, self.current_load)

    def update_pheromone(self, pheromone_matrix, evaporation_rate, deposit_rate):
        for i in range(len(self.route) - 1):
            start_location = self.route[i]
            end_location = self.route[i + 1]
            start_idx = self.location_index_mapping[start_location]
            end_idx = self.location_index_mapping[end_location]
            pheromone_matrix[start_idx][end_idx] *= (1 - evaporation_rate)
            pheromone_matrix[start_idx][end_idx] += deposit_rate / self.total_cost



def aco(start_location, end_locations, vehicles, distance_matrix, pheromone_matrix, simulation_folder, cluster, locations_df, alpha=ALPHA, beta= BETA, evaporation_rate=PHEROMONE_EVAPORATION_RATE, deposit_rate=PHEROMONE_DEPOSIT_RATE, num_ants=NUM_ANTS, iterations=100
):
    best_route = None
    best_cost = float('inf')
    current_vehicle = None
    best_distance = float('inf')
    total_fuel_consumed = float('inf')

    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}

    # Determine distribution centers based on start location
    if start_location.startswith('M'):
        distribution_centers = set()
        distribution_centers.add(start_location)
    elif start_location.startswith('W'):
        distribution_centers = set()
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
