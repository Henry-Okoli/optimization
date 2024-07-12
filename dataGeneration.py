import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic


folder =  'Output/WarehouseI'

# Load the data
df = pd.read_csv(folder + '.csv')

os.makedirs(folder,exist_ok=True)

# Function to calculate distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Get coordinates of M1 and M2 (Cores)
cores = df[df['code'].str.startswith('M')]
core_coords = {row['code']: (row['latitude'], row['longitude']) for _, row in cores.iterrows()}

# Calculate distances
distances = []

# Cores to all Warehouses
warehouses = df[df['code'].str.startswith('W')]
for core_code, core_coord in core_coords.items():
    for _, w in warehouses.iterrows():
        dist = calculate_distance(core_coord[0], core_coord[1], w['latitude'], w['longitude'])
        distances.append((core_code, w['code'], dist))

# Warehouses to Distribution Centers within each cluster
for cluster in df['ClusterCode'].unique():
    cluster_warehouses = warehouses[warehouses['ClusterCode'] == cluster]
    cluster_dcs = df[(df['ClusterCode'] == cluster) & (df['code'].str.startswith('D'))]
    
    for _, w in cluster_warehouses.iterrows():
        for _, dc in cluster_dcs.iterrows():
            dist = calculate_distance(w['latitude'], w['longitude'], dc['latitude'], dc['longitude'])
            distances.append((w['code'], dc['code'], dist))

# Distribution Centers to Retail Outlets within each cluster
for cluster in df['ClusterCode'].unique():
    cluster_dcs = df[(df['ClusterCode'] == cluster) & (df['code'].str.startswith('D'))]
    cluster_ros = df[(df['ClusterCode'] == cluster) & (df['code'].str.startswith('R'))]
    
    for _, dc in cluster_dcs.iterrows():
        for _, ro in cluster_ros.iterrows():
            dist = calculate_distance(dc['latitude'], dc['longitude'], ro['latitude'], ro['longitude'])
            distances.append((dc['code'], ro['code'], dist))

# Create a new DataFrame with the distances
distances_df = pd.DataFrame(distances, columns=['From', 'To', 'Distance'])

# Save distances DataFrame to CSV
distances_df.to_csv(folder + '/logistics_distances.csv', index=False)
print("Distances saved to 'logistics_distances.csv'")

# Create a graph
G = nx.Graph()

# Add edges to the graph
for _, row in distances_df.iterrows():
    G.add_edge(row['From'], row['To'], weight=row['Distance'])

# Custom position layout
pos = {}
layers = {'M': 0, 'W': 1, 'D': 2, 'R': 3}
clusters = df['ClusterCode'].unique()
node_counts = {cluster: {layer: 0 for layer in layers.values()} for cluster in clusters}
node_counts['Core'] = {0: 0}  # For core nodes

for node in G.nodes():
    layer = layers[node[0]]
    cluster = df[df['code'] == node]['ClusterCode'].iloc[0] if node[0] != 'M' else 'Core'
    
    if cluster == 'Core':
        x = 0
        node_counts['Core'][0] += 1
        y = node_counts['Core'][0]
    elif cluster == clusters[0]:  # cluster1
        x = 2 + layer  # Right side
        node_counts[cluster][layer] += 1
        y = node_counts[cluster][layer]
    else:  # cluster2
        x = -2 - layer  # Left side
        node_counts[cluster][layer] += 1
        y = node_counts[cluster][layer]
    
    pos[node] = (x, y)

# Adjust y-positions to center nodes vertically
max_counts = {cluster: max(counts.values()) for cluster, counts in node_counts.items()}
for node in G.nodes():
    x, y = pos[node]
    cluster = df[df['code'] == node]['ClusterCode'].iloc[0] if node[0] != 'M' else 'Core'
    y_adjusted = (y - (max_counts[cluster] + 1) / 2) / max_counts[cluster]
    pos[node] = (x, y_adjusted)

# Set node colors and sizes
color_map = {'M': 'red', 'W': 'green', 'D': 'blue', 'R': 'yellow'}
node_colors = [color_map[node[0]] for node in G.nodes()]

size_map = {'M': 1000, 'W': 800, 'D': 600, 'R': 400}
node_sizes = [size_map[node[0]] for node in G.nodes()]

# Plot the graph
plt.figure(figsize=(20, 15))
nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, font_size=8, font_weight='bold')

# Add edge labels with 3 decimal places
edge_labels = {(u, v): f'{d:.3f}' for (u, v, d) in G.edges(data='weight')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("Logistics Route Topology")
plt.axis('off')
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(folder + '/logistics_topology.png', dpi=300, bbox_inches='tight')
print("Topology plot saved to 'logistics_topology.png'")

# Close the plot to free up memory
plt.close()

print("Analysis complete. Files saved successfully.")