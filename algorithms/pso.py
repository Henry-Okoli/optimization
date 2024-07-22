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
RESUPPLY_THRESHOLD = 0.8  # Resupply when 80% of capacity is used (20% empty)

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
        
        self.max_route_length = len(end_locations) * 3  # Allow for some flexibility, but prevent infinite growth
        self.max_consecutive_dc_visits = 3
        self.total_cost = 0
        self.total_distance = 0
        self.total_fuel_consumed = 0
        self.resupply_threshold = RESUPPLY_THRESHOLD
        self.current_capacity = self.current_vehicle['Capacity_KG']
        self.position = self.initialize_position()
        self.velocity = self.initialize_velocity()
        self.best_position = self.position.copy()
        self.best_cost = float('inf')


    def initialize_position(self):
        route = [self.start_location]
        unserviced = list(self.end_locations)
        current_location = self.start_location
        current_load = 0  # Start with an empty vehicle
        consecutive_dc_visits = 0

        while unserviced and len(route) < self.max_route_length:
            if self.start_type == 'M' and len(route) == 1:
                next_location = min(unserviced, key=lambda loc: self.get_distance(current_location, loc))
            else:
                available_capacity = self.current_capacity - current_load
                if available_capacity <= self.resupply_threshold * self.current_capacity:
                    # Need to resupply
                    next_location = min(self.distribution_centers, key=lambda dc: self.get_distance(current_location, dc))
                    consecutive_dc_visits += 1
                else:
                    candidates = [loc for loc in unserviced if available_capacity >= self.location_demands[loc]]
                    if not candidates or consecutive_dc_visits >= self.max_consecutive_dc_visits:
                        next_location = min(unserviced, key=lambda loc: self.get_distance(current_location, loc))
                        consecutive_dc_visits = 0
                    else:
                        next_location = random.choice(candidates)
                        consecutive_dc_visits = 0

            route.append(next_location)
            if next_location in self.distribution_centers:
                current_load = 0  # Reset load after resupply
            else:
                current_load += self.location_demands[next_location]
                unserviced.remove(next_location)
            current_location = next_location

        if unserviced:
            print(f"Warning: Not all locations could be serviced. Unserviced: {unserviced}")

        return route


    def update_position(self):
        new_position = [self.start_location]
        remaining_locations = self.position[1:]
        current_load = 0  # Start with an empty vehicle
        consecutive_dc_visits = 0
        
        while remaining_locations and len(new_position) < self.max_route_length:
            available_capacity = self.current_capacity - current_load
            if available_capacity <= self.resupply_threshold * self.current_capacity:
                # Need to resupply
                next_location = min(self.distribution_centers, key=lambda dc: self.get_distance(new_position[-1], dc))
                consecutive_dc_visits += 1
            else:
                valid_locations = [loc for loc in remaining_locations if available_capacity >= self.location_demands.get(loc, 0)]
                if not valid_locations or consecutive_dc_visits >= self.max_consecutive_dc_visits:
                    valid_locations = remaining_locations  # Force visit to an end location
                
                if not valid_locations:
                    break  # No more locations can be visited
                
                # Use velocity to influence probabilities
                probabilities = [abs(self.velocity[self.position.index(loc)]) for loc in valid_locations]
                total_prob = sum(probabilities)
                if total_prob == 0:
                    next_location = random.choice(valid_locations)
                else:
                    probabilities = [p / total_prob for p in probabilities]
                    next_location = random.choices(valid_locations, weights=probabilities, k=1)[0]
                
                consecutive_dc_visits = 0
            
            new_position.append(next_location)
            if next_location in self.distribution_centers:
                current_load = 0  # Reset load after resupply
            else:
                current_load += self.location_demands.get(next_location, 0)
                remaining_locations.remove(next_location)
        
        self.position = new_position

    def evaluate(self):
        self.total_cost = 0
        self.total_distance = 0
        self.total_fuel_consumed = 0
        current_load = 0  # Start with an empty vehicle

        for i in range(len(self.position) - 1):
            start, end = self.position[i], self.position[i+1]
            distance = self.get_distance(start, end)
            self.total_distance += distance
            self.total_cost += calculate_cost([start, end], self.current_vehicle, self.distance_matrix, self.location_index_mapping, self.locations_df)
            self.total_fuel_consumed += calculate_fuel_consumption(distance, self.current_vehicle, current_load)

            if end in self.distribution_centers:
                current_load = 0  # Reset load after resupply
            else:
                current_load += self.location_demands.get(end, 0)

        # Check if all end locations were serviced
        unserviced = set(self.end_locations) - set(self.position)
        if unserviced:
            self.total_cost += len(unserviced) * 10000  # Large penalty for unserviced locations

        if self.total_cost < self.best_cost:
            self.best_cost = self.total_cost
            self.best_position = self.position.copy()

        return self.total_cost

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
            if i < len(self.velocity):
                new_vel = w * self.velocity[i] + cognitive + social
            else:
                new_vel = cognitive + social
            new_velocity.append(new_vel)
        
        self.velocity = new_velocity


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

    # Check if all locations can be served given the vehicle capacity
    max_demand = max(locations_df['Capacity_KG'])
  #  if max_demand > max(vehicle['Capacity_KG'] for vehicle in vehicles):
  #      raise ValueError("Vehicle capacity is insufficient to serve all locations")

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