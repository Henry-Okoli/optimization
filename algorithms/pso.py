
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
        self.end_locations = list(end_locations)  # Convert to list for indexing
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.location_index_mapping = location_index_mapping
        self.locations_df = locations_df
        self.distribution_centers = distribution_centers
        self.location_capacities = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.precomputed_distances = defaultdict(dict)
        self.current_vehicle = choose_vehicle(start_location, end_locations, vehicles)
        self.current_load = self.current_vehicle['Capacity_KG']
        self.location_demands = dict(zip(locations_df['code'], locations_df['Capacity_KG']))
        self.unserviced_locations = set(end_locations)
        self.start_type = start_location[0]
        self.end_type = end_locations[0][0] if end_locations else None

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
            cognitive = c1 * r1 * (self.position.index(self.best_position[i]) - self.position.index(self.position[i]))
            social = c2 * r2 * (self.position.index(global_best_position[i]) - self.position.index(self.position[i]))
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self):
        new_position = [self.start_location]
        remaining_locations = self.position[1:]
        
        for _ in range(len(self.position) - 1):
            if not remaining_locations:
                break
            probabilities = [abs(self.velocity[self.position.index(loc)]) for loc in remaining_locations]
            total_prob = sum(probabilities)
            if total_prob == 0:
                next_location = random.choice(remaining_locations)
            else:
                probabilities = [p / total_prob for p in probabilities]
                next_location = random.choices(remaining_locations, weights=probabilities, k=1)[0]
            new_position.append(next_location)
            remaining_locations.remove(next_location)
        
        self.position = new_position

    def evaluate(self):
        cost = calculate_cost(self.position, self.current_vehicle, self.distance_matrix, self.location_index_mapping)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_position = self.position.copy()
        return cost


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
    current_vehicle = choose_vehicle(start_location, end_locations, vehicles)
    best_distance = sum(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping) for i in range(len(best_route)-1))
    total_fuel_consumed = sum(calculate_fuel_consumption(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping), current_vehicle, current_vehicle['Capacity_KG']) for i in range(len(best_route)-1))

    return total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost
