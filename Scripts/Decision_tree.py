import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os

# Folder structure
data_dir = 'Preprocessed_data'

def load_dataset(data_dir):
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

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='Travel Day of Year')

    # Sort the DataFrame by 'Travel Day of Year' and 'Travel Hour'
    combined_df = combined_df.sort_values(by=['Travel Day of Year', 'Travel Hour'])



    dataset = combined_df.drop(columns=['Price ($)'])
    labels = combined_df['Price ($)']

    return dataset, labels

def evaluate_model(model, X_test, y_test):
    # Compute predictions
    y_pred = model.predict(X_test)

    # Compute MAE and MSE
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}\tMean Squared Error (MSE): {mse:.2f}")

    return mae, mse

def data_split(X_test, y_test):
    # [Short(below 800)]
    # JFK -> ATL(759)
    # JFK -> ORD(738)
    # ATL -> ORD(606)
    # ATL -> DFW(739)
    # DEN -> DFW(662)
    #
    # [Medium(800~1250)]
    # LAX -> DEN(860)
    # DEN -> ORD(806)
    # ORD -> DFW(801)
    # LAX -> DFW(1232)
    # ATL -> DEN(1196)
    #
    # [Long(over 1250)]
    # JFK -> LAX(2469)
    # JFK -> DFW(1388)
    # LAX -> DFW(1232)
    # LAX -> ORD(1741)
    # LAX -> ATL(1946)

    # 'JFK': 0, 'LAX': 1, 'DEN': 2, 'ATL': 3, 'DFW': 4, 'ORD': 5


    # Initialize empty DataFrames for each category
    short_distance_X = pd.DataFrame(columns=X_test.columns)
    short_distance_y = pd.Series()
    medium_distance_X = pd.DataFrame(columns=X_test.columns)
    medium_distance_y = pd.Series()
    long_distance_X = pd.DataFrame(columns=X_test.columns)
    long_distance_y = pd.Series()

    # Boolean indexing for short distance
    short_distance_conditions = (
        (X_test['Origin'] == 0) & (X_test['Destination'] == 3) |
        (X_test['Origin'] == 0) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 2) & (X_test['Destination'] == 4) |

        (X_test['Origin'] == 3) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 3) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 3) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 2)
    )
    short_distance_X = X_test[short_distance_conditions]
    short_distance_y = y_test[short_distance_conditions]

    # Boolean indexing for medium distance
    medium_distance_conditions = (
        (X_test['Origin'] == 1) & (X_test['Destination'] == 2) |
        (X_test['Origin'] == 2) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 2) |

        (X_test['Origin'] == 2) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 2) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 2) & (X_test['Destination'] == 3)
    )
    medium_distance_X = X_test[medium_distance_conditions]
    medium_distance_y = y_test[medium_distance_conditions]

    # Boolean indexing for long distance
    long_distance_conditions = (
        (X_test['Origin'] == 0) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 0) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 3) |

        (X_test['Origin'] == 1) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 1)
    )
    long_distance_X = X_test[long_distance_conditions]
    long_distance_y = y_test[long_distance_conditions]


    pd.set_option('display.max_columns', None)

    # Display the resulting DataFrames
    # print("Short Distance:")
    # print(short_distance_X.head())
    # print(short_distance_y.head())
    #
    # print("\nMedium Distance:")
    # print(medium_distance_X.head())
    # print(medium_distance_y.head())
    #
    # print("\nLong Distance:")
    # print(long_distance_X.head())
    # print(long_distance_y.head())

    overall_X = X_test
    overall_y = y_test

    X_test_dist = [short_distance_X, medium_distance_X, long_distance_X, overall_X]
    y_test_dist = [short_distance_y, medium_distance_y, long_distance_y, overall_y]


    return X_test_dist, y_test_dist


def main():
    # Load the dataset
    dataset, labels = load_dataset(data_dir)

    # Split the dataset into train, validation, and test sets
    X_train, X_other, y_train, y_other = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

    X_test_dist, y_test_dist = data_split(X_test, y_test)

    # Train the decision tree model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    mae_list = []
    mse_list = []
    distance_type = ["Short Distance", "Medium Distance", "Long Distance", "Overall"]

    # Evaluate the model for each distance type
    for i in range(len(X_test_dist)):
        X_test = X_test_dist[i]
        y_test = y_test_dist[i]

        # Evaluate the model
        print(distance_type[i])
        mae, mse = evaluate_model(model, X_test, y_test)

        mae_list.append(mae)
        mse_list.append(mse)

    # Print evaluation results
    print("\n\nEvaluation results")
    for i in range(len(distance_type)):
        print(f"Window {i+1} {distance_type[i]}: \tMAE: {mae_list[i]:.2f} \tMSE: {mse_list[i]:.2f}")

if __name__ == '__main__':
    main()
