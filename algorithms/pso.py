
import os
import pandas as pd
import random
from collections import defaultdict
from .helperfunctions  import choose_vehicle , getDistance
from .costfunctions import calculate_cost, calculate_fuel_consumption


# default values
NUM_ITERATIONS = 3
PARTICLE_POPULATION = 5  
INERTIA_WEIGHT = 0.7 
COGNITIVE_COEFFICIENT = 1.5 
SOCIAL_COEFFICIENT = 2.0




class Particle:
    def __init__(self, start_location, end_locations, vehicles, distance_matrix, location_index_mapping, locations_df, distribution_centers):
        self.start_location = start_location
        self.current_location = start_location
        self.end_locations = list(end_locations)  # Convert to list for indexing
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.location_index_mapping = location_index_mapping
        self.locations_df = locations_df
        self.distribution_centers = distribution_centers
        self.location_capacities = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.precomputed_distances = defaultdict(dict)
        self.current_vehicle = choose_vehicle(start_location, end_locations, vehicles, locations_df)
        self.current_load = self.current_vehicle['Capacity_KG']
        self.location_demands = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.unserviced_locations = set(end_locations)
        self.start_type = start_location[0]
        self.end_type = end_locations[0][0] if end_locations else None
        self.route = [start_location]
        self.position = self.initialize_position()
        self.velocity = self.initialize_velocity()
        self.best_position = self.position.copy()
        self.best_cost = float('inf')

    def initialize_position(self):
        return [self.start_location] + random.sample(self.end_locations, len(self.end_locations))

    def initialize_velocity(self):
        return [random.uniform(-1, 1) for _ in range(len(self.position))]

    def update_velocity(self, global_best_position, w, c1, c2):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            # cognitive = c1 * r1 * (self.position.index(self.best_position[i]) - self.position.index(self.position[i]))
            # social = c2 * r2 * (self.position.index(global_best_position[i]) - self.position.index(self.position[i]))
            #try:
            #    cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            #    social = c2 * r2 * (global_best_position[i] - self.position[i])
            #    self.velocity[i] = w * self.velocity[i] + cognitive + social
            #except IndexError as e:
            print("self.best_position:", self.best_position)
            print("self.position:", self.position)
            print("global_best_position:", global_best_position)
            print("Index:", i)
               # raise

    def update_position(self):
        new_position = [self.start_location]
        remaining_locations = self.position[1:]
        
        for _ in range(len(self.position) - 1):
            if not remaining_locations:
                break
            # probabilities = [abs(self.velocity[self.position.index(loc)]) for loc in remaining_locations]
            # total_prob = sum(probabilities)
            # if total_prob == 0:
            #    next_location = random.choice(remaining_locations)
            # else:
            #    probabilities = [p / total_prob for p in probabilities]
            #    next_location = random.choices(remaining_locations, weights=probabilities, k=1)[0]
            ####################################################################################################            
            if self.start_type == 'M' and self.current_location == self.start_location:
                next_location = self.force_end_location_visit()
            else:
                next_location = self.select_next_location()

            if next_location is None:
                print(f"No valid next location found. Breaking loop.")
                break

            if next_location in self.distribution_centers:
                self.route.append(next_location)
                new_position.append(next_location)
                self.current_load = self.current_vehicle['Capacity_KG']
                print(f"Restocked at DC {next_location}. Current load: {self.current_load}")
            else:
                required_load = self.location_demands[next_location]
                
                if self.current_load >= required_load:
                    self.route.append(next_location)
                    self.current_load -= required_load
                    self.unserviced_locations.remove(next_location)
                    new_position.append(next_location)
                    remaining_locations.remove(next_location)
                    print(f"Serviced {next_location}. Remaining load: {self.current_load}")
                else:
                    nearest_dc = self.find_nearest_dc()
                    if nearest_dc == self.current_location:
                        print(f"Already at nearest DC {nearest_dc}. Breaking loop to avoid infinite restocking.")
                        break
                    self.route.append(nearest_dc)
                    new_position.append(nearest_dc)
                    self.current_load = self.current_vehicle['Capacity_KG']
                    print(f"Insufficient load. Restocked at nearest DC {nearest_dc}")
            

            ####################################################################################################

        
        self.position = new_position

    def evaluate(self):
        cost = calculate_cost(self.position, self.current_vehicle, self.distance_matrix, self.location_index_mapping, self.locations_df)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_position = self.position.copy()
        return cost

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

    def find_nearest_dc(self):
        return min(self.distribution_centers, key=lambda dc: self.get_distance(self.current_location, dc))

    def force_end_location_visit(self):
        if self.end_locations:
            return min(self.end_locations, key=lambda loc: self.get_distance(self.current_location, loc))
        return self.find_nearest_dc()

    def get_distance(self, start, end):
        if start not in self.precomputed_distances or end not in self.precomputed_distances[start]:
            distance = getDistance(start, end, self.distance_matrix, self.location_index_mapping)
            self.precomputed_distances[start][end] = distance
            self.precomputed_distances[end][start] = distance
        return self.precomputed_distances[start][end]




def pso(start_location, end_locations, vehicles, distance_matrix, simulation_folder, cluster, locations_df, num_particles=PARTICLE_POPULATION, iterations=NUM_ITERATIONS):
    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}
    distribution_centers = set(locations_df[locations_df['code'].str.startswith('D')]['code'])
    distribution_centers.add(start_location)

    particles = [Particle(start_location, end_locations, vehicles, distance_matrix, location_index_mapping, locations_df, distribution_centers) for _ in range(num_particles)]

    global_best_position = None
    global_best_cost = float('inf')

    for iteration in range(iterations):
        for particle in particles:
            cost = particle.evaluate()
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position, INERTIA_WEIGHT, COGNITIVE_COEFFICIENT, SOCIAL_COEFFICIENT)
            particle.update_position()

        # Save data for this iteration
        iteration_pd = pd.DataFrame([[global_best_cost, global_best_position]], 
                                    columns=['total_cost', 'route'])
        iteration_pd.to_csv(os.path.join(simulation_folder, f'pso_iteration{iteration}.csv'), index=False)

    best_route = global_best_position
    best_cost = global_best_cost
    current_vehicle = choose_vehicle(start_location, end_locations, vehicles, locations_df)
    best_distance = sum(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping) for i in range(len(best_route)-1))
    total_fuel_consumed = sum(calculate_fuel_consumption(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping), current_vehicle, current_vehicle['Capacity_KG']) for i in range(len(best_route)-1))

    return total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost
