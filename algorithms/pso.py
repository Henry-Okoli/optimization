import os
import pandas as pd
import random
from collections import defaultdict
from .helperfunctions import choose_vehicle, getDistance
from .costfunctions import calculate_cost, calculate_fuel_consumption

# Default values
NUM_ITERATIONS = 100
PARTICLE_POPULATION = 10
INERTIA_WEIGHT = 0.7
COGNITIVE_COEFFICIENT = 1.5
SOCIAL_COEFFICIENT = 2.0

class Particle:
    def __init__(self, start_location, end_locations, vehicles, distance_matrix, location_index_mapping, locations_df, distribution_centers):
        self.start_location = start_location
        self.end_locations = set(end_locations)
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

        self.position = self.initialize_position()
        self.velocity = self.initialize_velocity()
        self.best_position = self.position.copy()
        self.best_cost = float('inf')
        self.total_cost = 0
        self.total_distance = 0
        self.total_fuel_consumed = 0

    def initialize_position(self):
        route = [self.start_location]
        unserviced = list(self.end_locations)
        current_location = self.start_location
        current_load = self.current_vehicle['Capacity_KG']

        while unserviced:
            if self.start_type == 'M' and len(route) == 1:
                next_location = min(unserviced, key=lambda loc: self.get_distance(current_location, loc))
            else:
                candidates = [loc for loc in unserviced if current_load >= self.location_demands[loc]] + list(self.distribution_centers)
                if not candidates:
                    next_location = min(self.distribution_centers, key=lambda dc: self.get_distance(current_location, dc))
                else:
                    next_location = random.choice(candidates)

            route.append(next_location)
            if next_location in self.distribution_centers:
                current_load = self.current_vehicle['Capacity_KG']
            else:
                current_load -= self.location_demands[next_location]
                unserviced.remove(next_location)
            current_location = next_location

        return route

    def initialize_velocity(self):
        return [random.uniform(-1, 1) for _ in range(len(self.position))]

    def update_velocity(self, global_best_position, w, c1, c2):
        new_velocity = []
        for i, location in enumerate(self.position):
            r1, r2 = random.random(), random.random()
            
            # Find the relative position of this location in the best positions
            personal_best_index = self.best_position.index(location) if location in self.best_position else i
            global_best_index = global_best_position.index(location) if location in global_best_position else i
            
            # Calculate the direction and magnitude of the velocity change
            cognitive = c1 * r1 * (personal_best_index - i)
            social = c2 * r2 * (global_best_index - i)
            
            # Update velocity
            new_vel = w * self.velocity[i] + cognitive + social
            new_velocity.append(new_vel)
        
        self.velocity = new_velocity

    def update_position(self):
        new_position = [self.start_location]
        remaining_locations = self.position[1:]
        current_load = self.current_vehicle['Capacity_KG']
        
        while remaining_locations:
            valid_locations = [loc for loc in remaining_locations if current_load >= self.location_demands.get(loc, 0)] + list(self.distribution_centers)
            
            if not valid_locations:
                next_location = min(self.distribution_centers, key=lambda dc: self.get_distance(new_position[-1], dc))
            else:
                # Use velocity to influence probabilities
                probabilities = [abs(self.velocity[self.position.index(loc)]) for loc in valid_locations]
                total_prob = sum(probabilities)
                if total_prob == 0:
                    next_location = random.choice(valid_locations)
                else:
                    probabilities = [p / total_prob for p in probabilities]
                    next_location = random.choices(valid_locations, weights=probabilities, k=1)[0]
            
            new_position.append(next_location)
            if next_location in self.distribution_centers:
                current_load = self.current_vehicle['Capacity_KG']
            else:
                current_load -= self.location_demands.get(next_location, 0)
                remaining_locations.remove(next_location)
        
        self.position = new_position

    def evaluate(self):
        self.total_cost = 0
        self.total_distance = 0
        self.total_fuel_consumed = 0
        current_load = self.current_vehicle['Capacity_KG']

        for i in range(len(self.position) - 1):
            start, end = self.position[i], self.position[i+1]
            distance = self.get_distance(start, end)
            self.total_distance += distance
            self.total_cost += calculate_cost([start, end], self.current_vehicle, self.distance_matrix, self.location_index_mapping, self.locations_df)
            self.total_fuel_consumed += calculate_fuel_consumption(distance, self.current_vehicle, current_load)

            if end in self.distribution_centers:
                current_load = self.current_vehicle['Capacity_KG']
            else:
                current_load -= self.location_demands.get(end, 0)

        if self.total_cost < self.best_cost:
            self.best_cost = self.total_cost
            self.best_position = self.position.copy()

        return self.total_cost

    def get_distance(self, start, end):
        if start not in self.precomputed_distances or end not in self.precomputed_distances[start]:
            distance = getDistance(start, end, self.distance_matrix, self.location_index_mapping)
            self.precomputed_distances[start][end] = distance
            self.precomputed_distances[end][start] = distance
        return self.precomputed_distances[start][end]

    

def pso(start_location, end_locations, vehicles, distance_matrix, simulation_folder, cluster, locations_df, num_particles=PARTICLE_POPULATION, iterations=NUM_ITERATIONS):
    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}
    
    if start_location.startswith('M'):
        distribution_centers = {start_location}
    elif start_location.startswith('W'):
        distribution_centers = {start_location}
    else:
        distribution_centers = set(locations_df[(locations_df['ClusterCode'] == cluster) & (locations_df['code'].str.startswith('D'))]['code'])

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

    best_particle = min(particles, key=lambda p: p.best_cost)
    best_route = best_particle.best_position
    best_cost = best_particle.best_cost
    best_distance = best_particle.total_distance
    total_fuel_consumed = best_particle.total_fuel_consumed

    return total_fuel_consumed, best_particle.current_vehicle, best_distance, best_route, best_cost


