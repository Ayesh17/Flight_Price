import os
from os.path import join
from src.google_flight_analysis.scrape import *


import os



# Ensure the 'extracted_data' folder exists
output_folder = os.path.join(os.getcwd(), '..', 'extracted_data')
os.makedirs(output_folder, exist_ok=True)

# Filter results we want format: origin, dest, origin_date, dest_date, ...
result = Scrape('JFK', 'LAX', '2024-03-20', '2025-03-22')

# Process the result using ScrapeObjects
ScrapeObjects(result)


# Display the data
print(result.data)

# Save the data to a CSV file in the 'extracted_data' folder
base_filename = 'flight_data.csv'
csv_file_path = os.path.join(output_folder, base_filename)

# Check if the file already exists
if os.path.isfile(csv_file_path):
    # If the file exists, find a new filename
    i = 1
    while True:
        new_filename = f'flight_data_{i}.csv'
        new_csv_file_path = os.path.join(output_folder, new_filename)
        if not os.path.isfile(new_csv_file_path):
            csv_file_path = new_csv_file_path
            break
        i += 1

# Save the data to the CSV file
result.data.to_csv(csv_file_path, index=False)

print(f"Data saved to: {csv_file_path}")
