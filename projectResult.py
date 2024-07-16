import os
import pandas as pd

# Define the base directory
base_dir = "output"  # Change this to your actual path

output_folder = f"{base_dir}/result"
os.makedirs(output_folder, exist_ok=True)

# Dictionary to hold the dataframes for each subdirectory
dataframes = []

# Walk through the directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith("simulation_data_") and file.endswith(".csv"):
            # Determine the subdirectory name
            subdirectory = os.path.relpath(root, base_dir)
            subdirectory_parts = subdirectory.split(os.sep)
            
            # Read the CSV file
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Add columns for each part of the subdirectory
            for i, part in enumerate(subdirectory_parts):
                df[f'subdirectory_level_{i+1}'] = part
            
            # Append the dataframe to the list
            dataframes.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Group the combined dataframe by subdirectory_level_1 and subdirectory_level_4
grouped = combined_df.groupby(['subdirectory_level_1', 'subdirectory_level_2'])

# Save the grouped dataframes to new CSV files
for (level_1, level_2), group in grouped:
    output_filename = f"{level_1}_{level_2}_combined_simulation_data.csv"
    output_path = os.path.join(output_folder, output_filename)
    group.to_csv(output_path, index=False)

print("Combining and grouping complete.")