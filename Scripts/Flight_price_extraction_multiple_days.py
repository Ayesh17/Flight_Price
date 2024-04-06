import os
import pandas as pd

from Flight_Price.src.google_flight_analysis.scrape import *

# Ensure the 'extracted_data' folder exists
output_folder = os.path.join(os.getcwd(), '..', 'extracted_data')
os.makedirs(output_folder, exist_ok=True)

#Airport codes
#JFK - New York
#LAX - Los Angeles
#DEN - Denver
#ATL -Atlanta
#DFW - Dalls
#ORD - Chicago

# Specify the origin, destination, and date range
origin = 'LAX'
dest = 'ATL'
start_date = '2024-04-05'  # Adjust as needed
end_date = '2024-08-17'  # Adjust as needed

# Gather results for each date within the range
scraped_data = []
for date in pd.date_range(start=start_date, end=end_date):
    print("Extracting data for flight on:", date)

    try:
        result = Scrape(dest, origin, date.strftime("%Y-%m-%d"))  # Format date as required by Scrape
        ScrapeObjects(result)
        scraped_data.append(result.data)
    except ValueError as e:
        print(f"Error for date {date}: {e}")

# Combine results into a single DataFrame
combined_data = pd.concat(scraped_data)

# Save the data to a CSV file in the 'extracted_data' folder
base_filename = "flight_data_range_" + origin + "_to_" + dest + ".csv"
csv_file_path = os.path.join(output_folder, base_filename)

# Check if the file already exists
if os.path.isfile(csv_file_path):
    # If the file exists, find a new filename
    i = 1
    while True:
        new_filename = "flight_data_range_" + origin + "_to_" + dest + "_" + str(i) + ".csv"
        new_csv_file_path = os.path.join(output_folder, new_filename)
        if not os.path.isfile(new_csv_file_path):
            csv_file_path = new_csv_file_path
            break
        i += 1

# Save the data to the CSV file
combined_data.to_csv(csv_file_path, index=False)

print(f"Data saved to: {csv_file_path}")
