import os
import time

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
origin = 'JFK'
dest = 'ATL'

start_date = '2024-04-05'  # Adjust as needed
end_date = '2024-08-17'  # Adjust as needed

start_date = '2024-04-17'  # Adjust as needed
end_date = '2024-09-17'  # Adjust as needed


# Set the number of retry attempts
max_retries = 10


# Gather results for each date within the range
scraped_data = []
for date in pd.date_range(start=start_date, end=end_date):
    print("Extracting data for flight on:", date)

    retries = 0
    while retries < max_retries:
        try:
            # Extract only the date part using strftime
            formatted_date = date.strftime("%Y-%m-%d")
            result = Scrape(dest, origin, formatted_date)
            # Call ScrapeObjects directly to avoid creating unnecessary objects
            ScrapeObjects(result)
            scraped_data.append(result.data)
            break  # Break out of the retry loop if scraping succeeds
        except ValueError as e:
            retries += 1
            print(f"Error scraping data for {date}. Retry {retries}/{max_retries}. Error: {e}")
            if retries == max_retries:
                print(f"Max retries reached for {date}. Skipping to the next date.")
                break  # Move to the next date if max retries reached
            time.sleep(1)  # Add a small delay between retries

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
