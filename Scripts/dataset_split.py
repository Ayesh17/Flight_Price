import os

import pandas as pd

# Load the CSV file
# Note: Ensure you update the file path location specified below on your end to read the CSV file.
file_path = "/Users/lawrencechuang/PycharmProjects/new_cs_791/Flight_Price/preprocessed_data/flight_data_range_DEN_to_DFW_preprocessed.csv"
df = pd.read_csv(file_path)

# Get the CSV file's filename
file_name = os.path.basename(file_path)

# Calculate the size of each dataset from the CSV file.
total_rows = len(df)
training_size = int(total_rows * 0.8)
testing_size = int(total_rows * 0.1)
validation_size = total_rows - training_size - testing_size

# Assign data to each dataset accordingly.
training_df = df.iloc[:training_size]
testing_df = df.iloc[training_size: training_size + testing_size]
validation_df = df.iloc[training_size + testing_size:]

# Save each dataset into a CSV file in its corresponding folder.
# Note: Ensure you update the file path location specified below on your end to generate the dataset.
training_df.to_csv(
    f"/Users/lawrencechuang/PycharmProjects/new_cs_791/Flight_Price/training_dataset/{file_name}_training_dataset.csv",
    index=False,
)
testing_df.to_csv(
    f"/Users/lawrencechuang/PycharmProjects/new_cs_791/Flight_Price/testing_dataset/{file_name}_testing_dataset.csv",
    index=False,
)
validation_df.to_csv(
    f"/Users/lawrencechuang/PycharmProjects/new_cs_791/Flight_Price/validation_dataset/{file_name}_validation_dataset.csv",
    index=False,
)

print("All datasets have been split into 3 categories!")
