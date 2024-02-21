# from src.google_flight_analysis.scrape import *
#---OR---#
import sys
sys.path.append('src/google_flight_analysis')
from scrape import *

# chain-trip format: origin, dest, date, origin, dest, date, ...
result = Scrape('JFK', 'LAX', '2024-03-20', '2024-03-22')

# Process the result using ScrapeObjects
ScrapeObjects(result)

# Display the data
print(result.data)

# Save the data to a CSV file
result.data.to_csv('flight_data.csv', index=False)
