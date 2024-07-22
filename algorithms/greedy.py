import random
from .helperfunctions import choose_vehicle, getDistance
from .costfunctions import calculate_cost, calculate_fuel_consumption

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
        self.current_vehicle = choose_vehicle(start_location, end_locations, vehicles, locations_df)
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
            if self.start_type == 'M' and len(route) == 1:
                next_location = self.force_end_location_visit(remaining_locations)
            elif self.current_load < 0.2 * self.current_vehicle['Capacity_KG']:
                next_location = self.find_nearest_dc(current_location)
            else:
                next_location = self.find_best_next_location(current_location, remaining_locations)

            if next_location is None:
                break

            route.append(next_location)
            
            if next_location in self.distribution_centers:
                self.current_load = self.current_vehicle['Capacity_KG']
            else:
                required_load = self.location_demands[next_location]
                self.current_load -= required_load
                remaining_locations.remove(next_location)

            current_location = next_location

            # Check for loops
            if len(route) > 3 and len(set(route[-3:])) == 2:
                if remaining_locations:
                    forced_location = random.choice(list(remaining_locations))
                    route.append(forced_location)
                    current_location = forced_location
                    self.current_load = max(0, self.current_load - self.location_demands[forced_location])
                    remaining_locations.remove(forced_location)

        cost = self.calculate_route_cost(route)
        return route, cost

    def force_end_location_visit(self, remaining_locations):
        return min(remaining_locations, key=lambda x: self.get_distance(self.start_location, x))

    def find_nearest_dc(self, current_location):
        return min((dc for dc in self.distribution_centers if dc != current_location), 
                   key=lambda dc: self.get_distance(current_location, dc))

    def find_best_next_location(self, current_location, remaining_locations):
        best_location = None
        best_score = float('-inf')

        for location in remaining_locations:
            if location == current_location:
                continue

            distance = self.get_distance(current_location, location)
            if distance == 0:
                continue

            required_load = self.location_demands[location]
            if self.current_load >= required_load:
                score = 1 / distance * 3  # Prioritize serviceable locations
            else:
                score = 1 / distance * 0.5  # Reduce priority for locations we can't fully service

            if score > best_score:
                best_score = score
                best_location = location

        return best_location

    def get_distance(self, loc1, loc2):
        return getDistance(loc1, loc2, self.distance_matrix, self.location_index_mapping)

    def calculate_route_cost(self, route):
        return calculate_cost(route, self.current_vehicle, self.distance_matrix, self.location_index_mapping, self.locations_df)

def greedy(start_location, end_locations, vehicles, distance_matrix, simulation_folder, cluster, locations_df):
    location_index_mapping = {code: idx for idx, code in enumerate(locations_df['code'])}
    distribution_centers = set(locations_df[locations_df['code'].str.startswith('D')]['code'])
    distribution_centers.add(start_location)

    greedy_solver = GreedySolver(start_location, end_locations, vehicles, distance_matrix, location_index_mapping, locations_df, distribution_centers)
    
    best_route, best_cost = greedy_solver.solve()

    current_vehicle = greedy_solver.current_vehicle
    best_distance = sum(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping) for i in range(len(best_route)-1))
    total_fuel_consumed = sum(calculate_fuel_consumption(getDistance(best_route[i], best_route[i+1], distance_matrix, location_index_mapping), current_vehicle, current_vehicle['Capacity_KG']) for i in range(len(best_route)-1))

    return total_fuel_consumed, current_vehicle, best_distance, best_route, best_cost