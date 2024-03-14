import os
import pandas as pd

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

        # Drop rows containing "min" or "avoids" in the "Airline" column
        df = df[~df['Airline(s)'].str.contains('min|avoids', case=False)]

        # Save the preprocessed data to a new CSV file in the output folder
        preprocessed_file_name = os.path.splitext(file_name)[0] + "_preprocessed.csv"
        output_file_path = os.path.join(output_folder, preprocessed_file_name)
        df.to_csv(output_file_path, index=False)

        print(f"Preprocessed data saved to: {output_file_path}")
