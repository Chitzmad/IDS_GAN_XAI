######################### find common features

import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("find_common_features.log")  # Output to file
    ]
)

# Input and output directory are same
# Common_features.csv will be produced in below directory
DIRECTORY = 'Summary'

def find_common_features(directory):
    try:
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        data = {}

        # Read each CSV file and store the selected features for each algorithm
        for file in csv_files:
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_csv(file_path)
                logging.info(f"Processing file: {file}")

                # Check if required columns are present
                if 'Algorithm' in df.columns and 'Selected Features' in df.columns:
                    for _, row in df.iterrows():
                        algorithm = row['Algorithm']
                        selected_features = set(row['Selected Features'].split(', '))  # Split by comma and space
                        data[algorithm] = selected_features
                else:
                    logging.warning(f"Required columns not found in file: {file}")
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue

        # Check if there are algorithms to compare
        if not data:
            logging.warning("No valid data found in the specified directory.")
            return

        # Prepare data for the output CSV file
        output_data = []
        algorithms = list(data.keys())

        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                try:
                    algo1, algo2 = algorithms[i], algorithms[j]
                    common_features = data[algo1].intersection(data[algo2])
                    output_data.append({
                        'algorithm1': algo1,
                        'algorithm2': algo2,
                        'count_common_features': len(common_features),
                        'common_features': ', '.join(common_features)
                    })
                    logging.info(f"Compared {algo1} and {algo2}: {len(common_features)} common features found.")
                except Exception as e:
                    logging.error(f"Error comparing {algo1} and {algo2}: {e}")

        # Write the output CSV file
        output_df = pd.DataFrame(output_data)
        output_file_path = os.path.join(directory, 'Common_features.csv')
        try:
            output_df.to_csv(output_file_path, index=False)
            logging.info(f"Common_features.csv has been created successfully at {output_file_path}")
        except Exception as e:
            logging.error(f"Error writing output CSV: {e}")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")

# Run the function
find_common_features(DIRECTORY)