import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic

folder =  'output/1W2DC'
filename = '1W2DC'

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

# Set node colors and sizes
color_map = {'M': 'red', 'W': 'green', 'D': 'blue', 'R': 'yellow'}
node_colors = []
node_sizes = []

# Place the core nodes (M)
core_nodes = sorted([node for node in G.nodes if node.startswith('M')])
core_x = 0
core_y_spacing = 10
for i, node in enumerate(core_nodes):
    pos[node] = (core_x, i * core_y_spacing)
    node_colors.append(color_map[node[0]])
    node_sizes.append(1000)

# Center the core nodes vertically
core_y_offset = 3 # (10 - len(core_nodes) * core_y_spacing) / 2
for i, node in enumerate(core_nodes):
    pos[node] = (core_x, i * core_y_spacing + core_y_offset)

# Place the warehouses (W) on the left or right based on their cluster, stacking vertically
for cluster in clusters:
    cluster_warehouses = sorted([node for node in G.nodes if node.startswith('W') and df[df['code'] == node]['ClusterCode'].iloc[0] == cluster])
    warehouse_x = 2 if cluster == clusters[0] else -2
    warehouse_y_spacing = 10 / len(cluster_warehouses) if len(cluster_warehouses) > 1 else 1
    for i, node in enumerate(cluster_warehouses):
        pos[node] = (warehouse_x, i * warehouse_y_spacing + 2.5)
        node_colors.append(color_map[node[0]])
        node_sizes.append(1000)

# Place the distribution centers (D), taking advantage of vertical space
for cluster in clusters:
    cluster_dcs = sorted([node for node in G.nodes if node.startswith('D') and df[df['code'] == node]['ClusterCode'].iloc[0] == cluster])
    dc_x = 3 if cluster == clusters[0] else -3
    dc_y_spacing = 10 / len(cluster_dcs) if len(cluster_dcs) > 1 else 1
    for i, node in enumerate(cluster_dcs):
        pos[node] = (dc_x, i * dc_y_spacing)
        node_colors.append(color_map[node[0]])
        node_sizes.append(1000)

# Place the retail outlets (R), taking advantage of vertical space
for cluster in clusters:
    cluster_ros = sorted([node for node in G.nodes if node.startswith('R') and df[df['code'] == node]['ClusterCode'].iloc[0] == cluster])
    ro_x = 4 if cluster == clusters[0] else -4
    ro_y_spacing = 10 / len(cluster_ros) if len(cluster_ros) > 1 else 1
    for i, node in enumerate(cluster_ros):
        pos[node] = (ro_x, i * ro_y_spacing)
        node_colors.append(color_map[node[0]])
        node_sizes.append(1000)

# Plot the graph
plt.figure(figsize=(12, 9))
nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, font_size=8, font_weight='bold')

# Add edge labels with 3 decimal places
edge_labels = {(u, v): f'{d:.3f}' for (u, v, d) in G.edges(data='weight')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title(f"{filename} - Logistics Route Topology")
plt.axis('off')
# plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(folder + '/logistics_topology.png', dpi=300, bbox_inches='tight')
print("Topology plot saved to 'logistics_topology.png'")

# Close the plot to free up memory
plt.close()

print("Analysis complete. Files saved successfully.")
