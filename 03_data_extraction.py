import pandas as pd

# Load the initial dataset
input_file = 'final_processed_CICI_updated.csv'
df = pd.read_csv(input_file)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Load the dataset containing algorithm names and features
algorithm_input_file = 'reduced_feature_set.xlsx'  # Replace with your actual dataset file name
algorithm_df = pd.read_excel(algorithm_input_file)

# Iterate over each row in the algorithm dataframe
for index, row in algorithm_df.iterrows():
    # Extract algorithm names and number of features
    algorithm1 = row[0]
    algorithm2 = row[1]
    number_of_features = row[2]
    
    # Extract the list of features from the last column and add 'label'
    features = row[-1].split(',')
    features.append('label')
    
    features = [feature.strip() for feature in features]
    # Filter the initial dataframe to keep only the specified columns
    filtered_df = df[features]

    # Define the output file name
    output_file = f"10data/{algorithm1}_{algorithm2}_{number_of_features}_selected_data.csv"
    
    # Save the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_file, index=False)

print("Data extraction completed successfully.")
