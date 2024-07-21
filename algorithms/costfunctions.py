from .helperfunctions import getDistance



def calculate_cost(route, vehicle, distance_matrix, location_index_mapping, locations_df):
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


# Maintenance cost calculation
def calculate_maintenance_cost(total_distance, vehicle):
    monthly_cost = vehicle['monthly_maintenance_cost']
    # return (total_distance / total_cycle_distance) * (monthly_cost / 4)
    return (monthly_cost / 4)



# Calculates the fuel consumed by the vehicle travelling a given distance carrying a given load capacity
def calculate_fuel_consumption(distance, vehicle, current_load):
    max_load = vehicle['Capacity_KG']
    fuel_max_load = vehicle['Fuel_Consumption_at_max_load_kmpg']
    fuel_zero_load = vehicle['Fuel_consumption_at_zero_load_kmpg']
    
    fuel_consumption = (
        (current_load / max_load) * fuel_max_load +
        (1 - current_load / max_load) * fuel_zero_load
    )
    return    distance / fuel_consumption

