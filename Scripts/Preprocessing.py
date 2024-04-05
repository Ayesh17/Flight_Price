import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

input_folder = os.path.join(os.getcwd(), '..', 'extracted_data')
output_folder = os.path.join(os.getcwd(), '..', 'preprocessed_data')

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all CSV files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        # Read the CSV file
        input_file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(input_file_path)
        print(df.head())

        df['Access Date'] = pd.to_datetime(df['Access Date'])
        df['Departure datetime'] = pd.to_datetime(df['Departure datetime'])
        df['Arrival datetime'] = pd.to_datetime(df['Arrival datetime'])

        # Get the overall time of the flight
        df['Time to Travel'] = (df['Arrival datetime'] - df[
            'Departure datetime']).dt.total_seconds() / 3600
        df['Time to Travel'] = df['Time to Travel'].round(2)  # Round the time to travel to two decimal points

        # Get the time difference from access date to travel date
        # Calculate the difference in days between 'Access Date' and 'Departure Date'
        # Extract date part from 'Departure datetime' column
        df['Departure Date'] = df['Departure datetime'].dt.date
        df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')
        df['Access-Departure Days'] = (df['Departure Date'] - df['Access Date']).dt.days

        # Extract date of the year (day of the year) and hour components
        # df['Travel Year'] = df['Departure datetime'].dt.year
        df['Travel Day of Year'] = df['Departure datetime'].dt.dayofyear
        df['Travel Hour'] = df['Departure datetime'].dt.hour
        df['Access Day of Year'] = df['Departure datetime'].dt.dayofyear

        # Drop instances where airline is wrong
        # Define a regular expression pattern to match non-letter characters or spaces
        pattern = r'[^a-zA-Z ]'
        # Filter the DataFrame to keep only rows where 'Airlines' column doesn't contain the pattern
        df = df[~df['Airline(s)'].str.contains(pattern)]

        # Drop the original 'Arrival datetime' and 'Departure datetime' columns
        df.drop(columns=['Arrival datetime'], inplace=True)
        df.drop(columns=['Departure datetime'], inplace=True)
        df.drop(columns=['Departure Date'], inplace=True)
        df.drop(columns=['Access Date'], inplace=True)
        df.drop(columns=['Travel Time'], inplace=True)
        df.drop(columns=['Layover'], inplace=True)
        df.drop(columns=['Airline(s)'], inplace=True)
        df.drop(columns=['CO2 Emission (kg)'], inplace=True)
        df.drop(columns=['Emission Diff (%)'], inplace=True)

        # convert non numerical values to numerical
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        df['Origin'] = label_encoder.fit_transform(df['Origin'])
        df['Destination'] = label_encoder.fit_transform(df['Destination'])

        print("df", df.head())

        column_names = df.columns.tolist()
        print("Column names:", column_names)

        # drop rows with incomplete data
        df.dropna(inplace=True)

        # Save the preprocessed data to a new CSV file in the output folder
        preprocessed_file_name = os.path.splitext(file_name)[0] + "_preprocessed.csv"
        output_file_path = os.path.join(output_folder, preprocessed_file_name)
        df.to_csv(output_file_path, index=False)

        print(f"Preprocessed data saved to: {output_file_path}")

