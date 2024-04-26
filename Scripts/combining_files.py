import os
import pandas as pd

# Folder structure
data_dir = 'Preprocessed_data'

# Get the current working directory
current_directory = os.getcwd()
base_directory = os.path.dirname(current_directory)
input_directory = os.path.join(base_directory, data_dir)

# List to store DataFrames
dfs = []

# Iterate over each file in the current directory
for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):  # Check if the file is a CSV file
        file_path = os.path.join(input_directory, file_name)  # Construct the file path
        df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
        dfs.append(df)  # Append the DataFrame to the list

# # Concatenate all DataFrames into a single DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# Concatenate all DataFrames into a single DataFrame based on 'travel_date' column
combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='Travel Day of Year')

# Sort the DataFrame by 'Travel Day of Year' and 'Travel Hour'
combined_df = combined_df.sort_values(by=['Travel Day of Year', 'Travel Hour'])

print("combined_df", combined_df.head())

column_names = combined_df.columns.tolist()

# Save as a CSV
combined_df.to_csv('combined_data.csv', index=False)