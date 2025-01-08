import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import pandas as pd

# Read the CSV file
df = pd.read_csv('CICIoT2023_part-00094_full_data.csv')

# Rename the last column to 'label'
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

# Save the modified DataFrame back to a CSV file (optional)
df.to_csv('updated_label_CICIoT2023_part.csv', index=False)
#df = pd.read_csv('merged_file_CICIDS_2017_Thursday-WorkingHours-Morning-Afternoon-WebAttacks.pcap_ISCXupdated_file.csv.csv')
print(df.head())



import pandas as pd
import numpy as np
import logging

# Set up logging to output to a file
logging.basicConfig(filename='error_log.txt', level=logging.INFO, format='%(message)s')



# Select only numeric columns to avoid issues with string data
numeric_df = df.select_dtypes(include=[np.number])

# Identify problematic values
infinity_mask = numeric_df.isin([np.inf, -np.inf])
threshold = np.finfo(np.float64).max
large_value_mask = numeric_df.abs() > threshold

# Combine masks for all problematic values
problematic_mask = infinity_mask | large_value_mask

# Log problematic values
if problematic_mask.any().any():
    logging.info("Problematic values found in 'input.csv':")
    problematic_indices = np.where(problematic_mask)
    for row, col in zip(*problematic_indices):
        logging.info(f"Row: {row}, Column: '{numeric_df.columns[col]}', Value: {numeric_df.iat[row, col]}")
else:
    logging.info("No infinity or extremely large values found in 'input.csv'.")

# Remove rows with any problematic values
# First, replace `inf`, `-inf`, and extremely large values with NaN, then drop rows with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_columns = df.select_dtypes(include=[np.number])
problematic_mask = (numeric_columns.abs() > threshold) | numeric_columns.isna()
rows_to_drop = problematic_mask.any(axis=1)  # Identify rows to drop

# Drop the identified rows
df_cleaned = df[~rows_to_drop]

# Save cleaned DataFrame to a new CSV file without problematic rows
df_cleaned.to_csv('CICIoT2023_cleaned.csv', index=False)
print("Data cleaning complete. Rows with problematic values have been removed. Output saved to 'CICIoT2023_cleaned.csv' and errors logged to 'error_log.txt'.")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df=df_cleaned

# Step 1: Remove null values
data = df.dropna()

features = df.drop('label', axis=1)
label = df['label']

# Handle non-numeric columns
encoder = LabelEncoder()
for col in features.select_dtypes(include=['object']).columns:
    features[col] = encoder.fit_transform(features[col])

# Step 2: Apply Min-Max Scaling to features
scaler = MinMaxScaler()
# features = data.drop('label', axis=1)  # Assuming 'label' is the target column
scaled_features = scaler.fit_transform(features)

# Step 3: Handle labels
labels = data['label']

# # If the labels are continuous and you want classification:
# if labels.dtype in ['float64', 'int64']:  # Check if labels are continuous
#     if len(labels.unique()) > 3:  # Ensure enough unique values for binning
#         labels = pd.qcut(labels, q=3, duplicates='drop')  # Automatically generate bins
#         num_bins = len(labels.cat.categories)  # Count the bins
#         # Dynamically assign category names
#         category_names = ['Category ' + str(i + 1) for i in range(num_bins)]
#         labels = labels.cat.rename_categories(category_names)
#     else:
#         # If there are not enough unique values, fallback to custom binning
#         max_label = labels.max()
#         min_label = labels.min()
#         labels = pd.cut(
#             labels, 
#             bins=[min_label, (min_label + max_label) / 2, max_label], 
#             labels=['low', 'high'], 
#             include_lowest=True
#         )

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 4: Combine scaled features and encoded labels
processed_data = pd.DataFrame(scaled_features, columns=features.columns)
processed_data['label'] = encoded_labels

# Step 5: Save the processed dataset to a CSV file
output_file_path = 'CICIoT2023_preprocessed.csv'
processed_data.to_csv(output_file_path, index=False)

print(f"Processed dataset has been saved to {output_file_path}")


import pandas as pd
from sklearn.model_selection import train_test_split


from datetime import datetime

start_time = datetime.now()


processed_data=pd.read_csv('CICIoT2023_preprocessed.csv')

Data_fraction = 0.1
MAX_ITER = 3



df=pd.read_csv('CICIoT2023_preprocessed.csv')
# Specify the class column name (the column you want to stratify by)
class_column = 'label'  # replace with your actual class column

# Split the data into a stratified sample and the rest, with the sample being 10% of the data
_, stratified_sample = train_test_split(df, test_size=0.1, stratify=df[class_column], random_state=1)

# Now 'stratified_sample' contains a 10% stratified sample of the original DataFrame
stratified_sample.to_csv('CICIoT2023_stratified.csv', index=False)

df=pd.read_csv('CICIoT2023_stratified.csv')
#df.head()
#df = processed_data.sample(frac=Data_fraction, random_state=42)

features = list(processed_data.columns[:-1])
# Round all numeric values to three decimal places
df = df.round(3)

df.to_csv('final_processed_merged_file_CICIDS_2017_Thursday-WorkingHours-Morning-Afternoon-WebAttacks.pcap_ISCX_updated_10percent.csv', index=False)

X = df[features].values
y = df.iloc[:, -1].values

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv
import time
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""**Cuckoo Search**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class CuckooSearch:
    def __init__(self, X, y, pop_size=20, max_iter=10, pa=0.25, Lambda=1.5):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pa = pa
        self.Lambda = Lambda
        self.num_features = X.shape[1]
        self.population = self.generate_initial_population()
        self.fitness_values = np.zeros(pop_size)
        self.features = None  # Can be set if feature names are available

    def levy_flight(self):
        u = np.random.normal(0, 1, size=1)
        v = np.random.normal(0, 1, size=1)
        step = u / np.power(np.abs(v), 1 / self.Lambda)
        return step

    def fitness_function(self, X_train, X_test, y_train, y_test, solution):
        selected_features = np.where(solution == 1)[0]
        selected_features = selected_features[selected_features < X_train.shape[1]]

        if len(selected_features) == 0:
            return 0

        model = KNeighborsClassifier()
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return accuracy_score(y_test, y_pred)

    def generate_initial_population(self):
        return np.random.randint(2, size=(self.pop_size, self.num_features))

    def get_best_solution(self):
        best_index = np.argmax(self.fitness_values)
        return self.population[best_index], self.fitness_values[best_index]

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        for i in range(self.pop_size):
            self.fitness_values[i] = self.fitness_function(X_train, X_test, y_train, y_test, self.population[i])

        best_solution, best_fitness = self.get_best_solution()

        for iteration in range(self.max_iter):
            new_population = self.population.copy()

            for i in range(self.pop_size):
                cuckoo = self.population[i] + self.levy_flight()
                cuckoo = np.clip(cuckoo, 0, 1) > np.random.random(self.num_features)

                fitness_cuckoo = self.fitness_function(X_train, X_test, y_train, y_test, cuckoo)

                if fitness_cuckoo > self.fitness_values[i]:
                    new_population[i] = cuckoo
                    self.fitness_values[i] = fitness_cuckoo

            abandon_indices = np.random.rand(self.pop_size) < self.pa
            new_population[abandon_indices] = self.generate_initial_population()[abandon_indices]

            for i in np.where(abandon_indices)[0]:
                self.fitness_values[i] = self.fitness_function(X_train, X_test, y_train, y_test, new_population[i])

            self.population = new_population
            best_solution, best_fitness = self.get_best_solution()

            print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

        selected_indices = np.where(best_solution == 1)[0]
        selected_features = [features[i] for i in selected_indices]
        return selected_features

"""**Evolutionary Programming**"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class EvolutionaryProgramming:
    def __init__(self, X, y, pop_size=20, max_iter=10, mutation_rate=0.1):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]

        if len(selected_features) == 0:
            return 1  # Return worst fitness if no features are selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)  # Simplified classifier
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        # Minimize the negative accuracy
        return -accuracy_score(y_test, y_pred)

    def mutate(self, solution):
        # Flip bits based on the mutation rate
        mutation = np.random.rand(len(solution)) < self.mutation_rate
        return np.where(mutation, 1 - solution, solution)

    def search(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.randint(0, 2, size=(self.pop_size, self.X.shape[1]))

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # Evolutionary Programming Loop
        for generation in range(self.max_iter):
            # Evaluate fitness of the current population
            fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

            # Select parents (top 50% individuals)
            num_parents = self.pop_size // 2
            sorted_indices = np.argsort(fitness_scores)
            parents = population[sorted_indices[:num_parents]]

            # Mutate offspring
            offspring = np.array([self.mutate(parents[np.random.randint(num_parents)]) for _ in range(self.pop_size)])

            # Combine parents and offspring to form the new population
            population = np.vstack((parents, offspring))

        # End time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for optimization: {total_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

"""**Firefly Optimization**"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class Firefly:
    def __init__(self, X, y, pop_size=20, max_iter=10, alpha=0.2, beta_min=0.2, gamma=1.0):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]

        if len(selected_features) == 0:
            return 1  # Return worst fitness if no features are selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)  # Simplified classifier
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        # Minimize the negative accuracy
        return -accuracy_score(y_test, y_pred)

    def move_firefly(self, firefly_i, firefly_j, beta):
        random_factor = self.alpha * (np.random.rand(len(firefly_i)) - 0.5)
        new_position = firefly_i + beta * (firefly_j - firefly_i) + random_factor
        return np.clip(new_position, 0, 1)  # Ensure solution is in [0, 1]

    def search(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.rand(self.pop_size, self.X.shape[1])

        # Compute initial fitness for the population
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # Firefly Optimization Loop
        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness_scores[j] < fitness_scores[i]:  # Firefly j is more attractive
                        beta = self.beta_min * np.exp(-self.gamma * np.linalg.norm(population[i] - population[j]) ** 2)
                        population[i] = self.move_firefly(population[i], population[j], beta)

                        # Recalculate fitness after moving
                        fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        # End time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for optimization: {total_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected feature names from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]
        selected_features = [features[i] for i in selected_indices]
        return selected_features

"""**Adaptive Bacterial Foraging Optimization**"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class AdaptiveBacterialForaging:
    def __init__(self, X, y, pop_size=20, max_iter=10, C=0.1, elimination_prob=0.1, reproduction_prob=0.5):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.C = C
        self.elimination_prob = elimination_prob
        self.reproduction_prob = reproduction_prob
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]

        if len(selected_features) == 0:
            return 1  # Return worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        # Minimize the negative accuracy
        return -accuracy_score(y_test, y_pred)

    def chemotaxis(self, bacteria, fitness_scores, X_train, X_test, y_train, y_test):
        for i in range(len(bacteria)):
            step = self.C * np.random.randn(bacteria.shape[1])
            new_bacteria = bacteria[i] + step
            new_bacteria = np.clip(new_bacteria, 0, 1)  # Ensure solution remains in [0, 1]

            # Calculate fitness for the new solution
            new_fitness = self.fitness_function(new_bacteria, X_train, X_test, y_train, y_test)
            if new_fitness < fitness_scores[i]:  # If fitness improves, update bacteria position
                bacteria[i] = new_bacteria
                fitness_scores[i] = new_fitness

        return bacteria, fitness_scores

    def reproduction(self, bacteria, fitness_scores):
        # Sort bacteria by fitness and select the better half
        sorted_indices = np.argsort(fitness_scores)
        bacteria = bacteria[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        # Replace the worst half by cloning the better half
        for i in range(len(bacteria) // 2):
            bacteria[-(i+1)] = bacteria[i]
            fitness_scores[-(i+1)] = fitness_scores[i]

        return bacteria, fitness_scores

    def elimination_dispersal(self, bacteria, fitness_scores, X_train, X_test, y_train, y_test):
        for i in range(len(bacteria)):
            if np.random.rand() < self.elimination_prob:
                # Replace the bacteria with a new random solution
                bacteria[i] = np.random.rand(bacteria.shape[1])
                fitness_scores[i] = self.fitness_function(bacteria[i], X_train, X_test, y_train, y_test)

        return bacteria, fitness_scores

    def search(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        bacteria = np.random.rand(self.pop_size, self.X.shape[1])

        # Compute initial fitness for the population
        fitness_scores = np.array([self.fitness_function(bac, X_train, X_test, y_train, y_test) for bac in bacteria])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # ABFO Loop
        for generation in range(self.max_iter):
            # Chemotaxis
            bacteria, fitness_scores = self.chemotaxis(bacteria, fitness_scores, X_train, X_test, y_train, y_test)

            # Reproduction
            if np.random.rand() < self.reproduction_prob:
                bacteria, fitness_scores = self.reproduction(bacteria, fitness_scores)

            # Elimination and Dispersal
            bacteria, fitness_scores = self.elimination_dispersal(bacteria, fitness_scores, X_train, X_test, y_train, y_test)

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        # End time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for optimization: {total_time:.2f} seconds")

        # Return the names of the best selected features
        best_solution = bacteria[np.argmin(fitness_scores)]
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class AntColony:
    def __init__(self, X, y, pop_size=20, max_iter=10, alpha=1.0, beta=1.0, decay=0.1):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]

        if len(selected_features) == 0:
            return 1  # Return worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        # Minimize the negative accuracy
        return -accuracy_score(y_test, y_pred)

    # Function to initialize pheromone matrix
    def initialize_pheromone_matrix(self, initial_pheromone=0.1):
        return np.ones(self.features) * initial_pheromone

    # Function to choose features based on pheromone values
    def select_features(self, pheromone):
        probabilities = pheromone ** self.alpha
        probabilities /= np.sum(probabilities)
        return np.random.rand(len(pheromone)) < probabilities

    # Function to update pheromone matrix
    def update_pheromone(self, pheromone, best_solution):
        pheromone *= (1 - self.decay)  # Evaporation
        pheromone += best_solution  # Reinforce pheromone on the best solution
        return pheromone

    # Ant Colony Optimization for feature selection
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize pheromone matrix for all features
        pheromone = self.initialize_pheromone_matrix()

        # Start time to monitor the timing of each generation
        start_time = time.time()

        best_solution = None
        best_fitness = float('inf')

        # ACO Loop
        for generation in range(self.max_iter):
            population = np.zeros((self.pop_size, self.features))
            fitness_scores = np.zeros(self.pop_size)

            # Each ant constructs a solution
            for i in range(self.pop_size):
                # Ant selects features based on pheromone trail
                solution = self.select_features(pheromone)
                population[i] = solution

                # Calculate fitness for the constructed solution
                fitness_scores[i] = self.fitness_function(solution, X_train, X_test, y_train, y_test)

                # Update best solution if necessary
                if fitness_scores[i] < best_fitness:
                    best_fitness = fitness_scores[i]
                    best_solution = solution

            # Update pheromone matrix based on the best solution found in this iteration
            pheromone = self.update_pheromone(pheromone, best_solution)

            # Track progress
            print(f"Generation {generation_counter}: Best fitness = {-best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for optimization: {total_time:.2f} seconds")

        selected_indices = np.where(best_solution > 0.5)[0]
        selected_features = [features[i] for i in selected_indices]
        return selected_features

"""**Artificial Bee Colony Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class ArtificialBeeColony:
    def __init__(self, X, y, pop_size=20, max_iter=10, limit=5):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]

        if len(selected_features) == 0:
            return 1  # Return worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        # Minimize the negative accuracy
        return -accuracy_score(y_test, y_pred)

    # ABC Optimization process
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.rand(self.pop_size, self.features)

        # Initialize fitness scores
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # ABC Optimization Loop
        for generation in range(self.max_iter):
            # Employed bees search for new solutions
            for i in range(self.pop_size):
                # Choose a random feature to change
                new_solution = population[i].copy()
                random_feature = np.random.randint(0, self.features)
                new_solution[random_feature] = 1 - new_solution[random_feature]  # Flip the feature selection

                # Evaluate the new solution
                new_fitness = self.fitness_function(new_solution, X_train, X_test, y_train, y_test)

                # Greedily replace the old solution if the new one is better
                if new_fitness < fitness_scores[i]:
                    population[i] = new_solution
                    fitness_scores[i] = new_fitness

            # Onlooker bees select solutions based on fitness
            probabilities = 1 / (1 + fitness_scores)
            probabilities /= np.sum(probabilities)  # Normalize probabilities

            for i in range(self.pop_size):
                if np.random.rand() < probabilities[i]:  # Select this solution
                    new_solution = population[i].copy()
                    random_feature = np.random.randint(0, self.features)
                    new_solution[random_feature] = 1 - new_solution[random_feature]  # Flip the feature selection

                    # Evaluate the new solution
                    new_fitness = self.fitness_function(new_solution, X_train, X_test, y_train, y_test)

                    # Greedily replace if the new one is better
                    if new_fitness < fitness_scores[i]:
                        population[i] = new_solution
                        fitness_scores[i] = new_fitness

            # Scout bees search for new solutions if no improvement after limit iterations
            for i in range(self.pop_size):
                if fitness_scores[i] >= self.limit:  # Check if it meets the limit
                    population[i] = np.random.rand(self.features)  # Restart solution
                    fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for optimization: {total_time:.2f} seconds")

        # Return the best solution found
        best_solution = population[np.argmin(fitness_scores)]
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Sine Cosine Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class SineCosine:
    def __init__(self, X, y, pop_size=20, max_iter=10, a=2):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.a = a
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]

        if len(selected_features) == 0:
            return 1  # Return worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        # Minimize the negative accuracy
        return -accuracy_score(y_test, y_pred)

    # Sine Cosine Optimization process
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.rand(self.pop_size, self.features)

        # Compute initial fitness for the population
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # SCO Optimization Loop
        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                # Generate a new solution
                new_solution = np.zeros_like(population[i])

                for j in range(self.features):
                    # Calculate the sine and cosine components for the feature
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    new_solution[j] = (np.sin(r1 * np.pi) * population[i][j] +
                                       np.cos(r2 * np.pi) * (np.mean(population[:, j]) - population[i][j]))

                    # Ensure the solution is in [0, 1]
                    new_solution[j] = np.clip(new_solution[j], 0, 1)

                # Evaluate the new solution
                new_fitness = self.fitness_function(new_solution, X_train, X_test, y_train, y_test)

                # Greedily replace the old solution if the new one is better
                if new_fitness < fitness_scores[i]:
                    population[i] = new_solution
                    fitness_scores[i] = new_fitness

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for optimization: {total_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Social Spider Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class SocialSpider:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        return -accuracy_score(y_test, y_pred)  # Minimize the negative accuracy

    # Social Spider Optimization process
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.rand(self.pop_size, self.features)

        # Compute initial fitness for the population
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # SSO Optimization Loop
        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                spider_fitness = fitness_scores[i]
                best_spider = np.argmin(fitness_scores)

                # Update position based on the best spider
                for j in range(self.features):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    if np.random.rand() < 0.5:
                        new_value = population[i][j] + r1 * (population[best_spider][j] - population[i][j])
                    else:
                        new_value = population[i][j] + r2 * (np.mean(population, axis=0)[j] - population[i][j])

                    # Ensure the new value is within [0, 1]
                    population[i][j] = np.clip(new_value, 0, 1)

                # Evaluate the new solution
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Symbiotic Organisms Search Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class Symbiotic:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        return -accuracy_score(y_test, y_pred)  # Minimize the negative accuracy

    # Symbiotic Organisms Search Optimization process
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.rand(self.pop_size, self.features)

        # Compute initial fitness for the population
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # SOS Optimization Loop
        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                selected_index = np.random.choice(self.pop_size)
                while selected_index == i:
                    selected_index = np.random.choice(self.pop_size)

                # Mimic symbiotic behavior: adjust current organism towards a better neighbor
                population[i] += np.random.rand(self.features) * (population[selected_index] - population[i])

                # Clip to ensure values are within bounds
                population[i] = np.clip(population[i], 0, 1)

                # Evaluate the updated solution
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class BacterialForaging:
    def __init__(self, X, y, pop_size=20, max_iter=10, num_steps=10, step_size=0.1):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.num_steps = num_steps
        self.step_size = step_size
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Worst fitness if no features selected

        # Train and test classifier with the selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        return -accuracy_score(y_test, y_pred)  # Minimize the negative accuracy

    # Bacterial Foraging Optimization process
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population (random binary solutions)
        population = np.random.rand(self.pop_size, self.features)

        # Compute initial fitness for the population
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # BFO Optimization Loop
        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                for step in range(self.num_steps):
                    # Randomly adjust the bacterium's position
                    previous_position = population[i].copy()
                    population[i] += (np.random.rand(self.features) - 0.5) * self.step_size

                    # Clip to ensure values are within bounds
                    population[i] = np.clip(population[i], 0, 1)
                    new_fitness = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

                    # If new fitness is better, keep the new position; else, revert
                    if new_fitness < fitness_scores[i]:
                        fitness_scores[i] = new_fitness
                    else:
                        population[i] = previous_position  # Revert to previous position

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)

        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Bat Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class Bat:
    def __init__(self, X, y, pop_size=20, max_iter=10, alpha=0.9, gamma=1.0):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.features = X.shape[1]

    # Fitness function for feature selection
    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Worst fitness if no features selected

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])

        return -accuracy_score(y_test, y_pred)  # Minimize the negative accuracy

    # Bat Optimization process
    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize population and velocities
        population = np.random.rand(self.pop_size, self.features)
        velocities = np.zeros_like(population)

        # Compute initial fitness
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the timing of each generation
        start_time = time.time()

        # Bat Optimization Loop
        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.rand()
                if r > 0.5:
                    # Adjust the velocity and update the solution
                    velocities[i] += (population[np.random.randint(self.pop_size)] - population[i]) * np.random.rand()
                    population[i] += velocities[i]

                # Ensure values are within [0, 1]
                population[i] = np.clip(population[i], 0, 1)

                # Calculate the fitness of the new solution
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            # Track progress
            best_fitness = -np.min(fitness_scores)
            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        # Return the best solution found
        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)

        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Big Bang Big Crunch**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class BigBangBigCrunch:
    def __init__(self, X, y, pop_size=20, max_iter=10, explosion_rate=0.3):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.explosion_rate = explosion_rate
        self.features = X.shape[1]

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Worst fitness if no features are selected

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)  # Minimize the negative accuracy

    def search(self):
        global generation_counter
        generation_counter = 0

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the population and compute initial fitness
        population = np.random.rand(self.pop_size, self.features)
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        # Start time to monitor the duration of the optimization
        start_time = time.time()

        # Big Bang-Big Crunch Optimization Loop
        for generation in range(self.max_iter):
            best_fitness = -np.min(fitness_scores)
            average_fitness = np.mean(fitness_scores)

            # Big Bang (explosion): Randomly initialize new population
            if np.random.rand() < self.explosion_rate:
                population = np.random.rand(self.pop_size, self.features)
                fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

            # Big Crunch: Move population toward the center of mass based on fitness
            else:
                for i in range(self.pop_size):
                    population[i] += (np.random.rand(self.features) - 0.5) * (average_fitness - fitness_scores[i])
                    population[i] = np.clip(population[i], 0, 1)
                    fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            print(f"Generation {generation_counter}: Best fitness = {best_fitness}")
            generation_counter += 1

        # End time
        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        # Find and return the best solution found
        best_solution_index = np.argmin(fitness_scores)
        best_solution = population[best_solution_index]
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Biogeography-based Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class Biogeography:
    def __init__(self, X, y, pop_size=20, max_iter=10, migration_rate=0.3):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.migration_rate = migration_rate
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Worst fitness if no features are selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            best_fitness = -np.min(fitness_scores)
            average_fitness = np.mean(fitness_scores)

            for i in range(self.pop_size):
                if np.random.rand() < self.migration_rate:
                    # Migrate features from a better solution
                    donor_index = np.random.choice(np.flatnonzero(fitness_scores == np.min(fitness_scores)))
                    population[i] = population[donor_index] + np.random.normal(0, 0.1, size=self.X.shape[1])
                    population[i] = np.clip(population[i], 0, 1)
                    fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        # Extract the selected features from the best solution and return them
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Tug of War Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class TugOfWar:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Assign worst fitness if no features are selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    # Random mutation to introduce diversity
                    population[i] = np.random.rand(self.X.shape[1])
                else:
                    # Update based on the best solution
                    best_index = np.argmin(fitness_scores)
                    population[i] = population[best_index] + np.random.normal(0, 0.1, size=self.X.shape[1])
                    population[i] = np.clip(population[i], 0, 1)

                # Recalculate fitness for the updated individual
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        # Extract the selected features from the best solution
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

"""**Water Cycle Optimization**"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class WaterCycle:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Assign worst fitness if no features are selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    # Water movement - introduce randomness
                    population[i] = np.random.rand(self.X.shape[1])
                else:
                    # Move towards the best solution (simulating flow towards an optimal solution)
                    best_index = np.argmin(fitness_scores)
                    population[i] = population[best_index] + np.random.normal(0, 0.1, size=self.X.shape[1])
                    population[i] = np.clip(population[i], 0, 1)

                # Update fitness for the new solution
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class WhaleOptimization:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Assign worst fitness if no features are selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                best_index = np.argmin(fitness_scores)
                r = np.random.rand()
                A = 2 * np.random.rand() - 1
                C = 2 * np.random.rand()

                if r < 0.5:
                    population[i] = population[best_index] - A * np.abs(C * population[best_index] - population[i])
                else:
                    population[i] = population[best_index] + A * np.abs(C * population[best_index] - population[i])

                # Ensure values remain in the range [0, 1]
                population[i] = np.clip(population[i], 0, 1)
                # Update fitness score for the new solution
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class WhaleSwarmOptimization:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Assign worst fitness if no features are selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.rand()
                if r < 0.5:
                    best_index = np.argmin(fitness_scores)
                    A = np.random.rand()
                    population[i] = population[best_index] + A * np.abs(population[best_index] - population[i])
                else:
                    worst_index = np.argmax(fitness_scores)
                    A = np.random.rand()
                    population[i] = population[worst_index] - A * np.abs(population[worst_index] - population[i])

                # Ensure values remain in the range [0, 1]
                population[i] = np.clip(population[i], 0, 1)
                # Update fitness score for the new solution
                fitness_scores[i] = self.fitness_function(population[i], X_train, X_test, y_train, y_test)

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        return [features[i] for i in np.where(best_solution > 0.5)[0]]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class CatSwarmOptimizer:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    # Exploration phase
                    self.population[i] = np.random.rand(self.X.shape[1])
                else:
                    # Exploitation phase
                    best_index = np.argmin(fitness_scores)
                    self.population[i] = self.population[best_index] + np.random.normal(0, 0.1, size=self.X.shape[1])
                    self.population[i] = np.clip(self.population[i], 0, 1)  # Ensure values stay within [0, 1]

                # Update fitness score for the new solution
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class ChickenSwarmOptimizer:
    def __init__(self, X, y , pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    # Exploration phase
                    self.population[i] = np.random.rand(self.X.shape[1])
                else:
                    # Exploitation phase
                    best_index = np.argmin(fitness_scores)
                    self.population[i] = self.population[best_index] + np.random.normal(0, 0.1, size=self.X.shape[1])
                    self.population[i] = np.clip(self.population[i], 0, 1)  # Ensure values stay within [0, 1]

                # Update fitness score for the new solution
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class ClonalSelectionOptimizer:
    def __init__(self, X, y , pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            best_index = np.argmin(fitness_scores)  # Index of the best solution
            for i in range(self.pop_size):
                # Cloning and mutation
                if i == best_index:
                    self.population[i] += np.random.normal(0, 0.1, size=self.X.shape[1])
                else:
                    self.population[i] += np.random.rand(self.X.shape[1]) * (self.population[best_index] - self.population[i])

                self.population[i] = np.clip(self.population[i], 0, 1)  # Ensure values stay within [0, 1]
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class CoralReefsOptimizer:
    def __init__(self, X, y , pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            best_index = np.argmin(fitness_scores)  # Index of the best solution
            for i in range(self.pop_size):
                # Update positions based on the best coral
                self.population[i] = self.population[best_index] + np.random.normal(0, 0.1, size=self.X.shape[1])
                self.population[i] = np.clip(self.population[i], 0, 1)  # Ensure values stay within [0, 1]

                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class FireworkOptimization:
    def __init__(self, X, y , pop_size=20, max_iter=10, explosion_strength=0.2):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.explosion_strength = explosion_strength
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                # Generate sparks based on the fitness score
                num_sparks = int(self.pop_size * (1 - fitness_scores[i]))
                sparks = np.array([self.population[i] + np.random.normal(0, self.explosion_strength, size=self.X.shape[1]) for _ in range(num_sparks)])

                for spark in sparks:
                    spark = np.clip(spark, 0, 1)  # Ensure sparks stay within [0, 1]
                    fitness = self.fitness_function(spark)

                    if fitness < fitness_scores[i]:
                        self.population[i] = spark
                        fitness_scores[i] = fitness

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class FlowerPollination:
    def __init__(self, X, y , pop_size=20, max_iter=10, p=0.8):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.p = p
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < self.p:
                    # Global pollination
                    best_index = np.argmin(fitness_scores)
                    self.population[i] += np.random.normal(0, 0.1, size=self.X.shape[1]) * (self.population[best_index] - self.population[i])
                else:
                    # Local pollination
                    j = np.random.randint(self.pop_size)
                    self.population[i] += np.random.normal(0, 0.1, size=self.X.shape[1]) * (self.population[j] - self.population[i])

                self.population[i] = np.clip(self.population[i], 0, 1)
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class GravitationalSearch:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            total_fitness = np.sum(fitness_scores)
            gravitational_force = (fitness_scores / total_fitness).reshape(-1, 1)
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness_scores[j] < fitness_scores[i]:  # Attractive force
                        force = gravitational_force[j] / np.linalg.norm(self.population[i] - self.population[j])
                        self.population[i] += force * (self.population[j] - self.population[i])

                self.population[i] = np.clip(self.population[i], 0, 1)
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class GrayWolfOptimization:
    def __init__(self, X, y, features, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.features = features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            alpha_index = np.argmin(fitness_scores)
            alpha = self.population[alpha_index]
            a = 2 - generation * (2 / self.max_iter)

            for i in range(self.pop_size):
                for j in range(3):  # Update using alpha, beta, and delta wolves
                    r = np.random.rand(self.X.shape[1])
                    A = 2 * a * r - a
                    C = 2 * np.random.rand(self.X.shape[1])
                    D = np.abs(C * alpha - self.population[i])
                    self.population[i] = alpha - A * D

                self.population[i] = np.clip(self.population[i], 0, 1)
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class GreenHeronsOptimization:
    def __init__(self, X, y, pop_size=20, max_iter=10, p=0.5):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.p = p
        self.generation_counter = 0
        self.population = np.random.rand(pop_size, X.shape[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < self.p:
                    # Update by foraging behavior
                    self.population[i] += np.random.normal(0, 0.1, size=self.X.shape[1]) * (np.random.rand() * (self.population.max(axis=0) - self.population[i]))
                else:
                    # Update by avoiding predators
                    self.population[i] += np.random.normal(0, 0.1, size=self.X.shape[1]) * (np.random.rand() * (self.population.min(axis=0) - self.population[i]))

                self.population[i] = np.clip(self.population[i], 0, 1)
                fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class GreyWolfOptimizer:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            alpha_index = np.argmin(self.fitness_scores)
            beta_index = np.argsort(self.fitness_scores)[1]
            delta_index = np.argsort(self.fitness_scores)[2]

            alpha = self.population[alpha_index]
            beta = self.population[beta_index]
            delta = self.population[delta_index]

            a = 2 - generation * (2 / self.max_iter)

            for i in range(self.pop_size):
                r1 = np.random.rand(self.X.shape[1])
                r2 = np.random.rand(self.X.shape[1])
                r3 = np.random.rand(self.X.shape[1])

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha - self.population[i])
                self.population[i] = alpha - A1 * D_alpha

                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta - self.population[i])
                self.population[i] = beta - A2 * D_beta

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta - self.population[i])
                self.population[i] = delta - A3 * D_delta

                self.population[i] = np.clip(self.population[i], 0, 1)
                self.fitness_scores[i] = self.fitness_function(self.population[i])

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class HarmonySearch:
    def __init__(self, X, y, pop_size=20, max_iter=10, harmony_memory_size=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.harmony_memory_size = harmony_memory_size
        self.generation_counter = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.harmony_memory = np.random.rand(harmony_memory_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.harmony_memory])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            new_harmony = np.zeros(self.X.shape[1])
            for i in range(self.X.shape[1]):
                if np.random.rand() < 0.5:  # Choose from harmony memory
                    idx = np.random.randint(0, self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                else:  # Random choice
                    new_harmony[i] = np.random.rand()

            new_harmony = np.clip(new_harmony, 0, 1)
            new_fitness = self.fitness_function(new_harmony)

            # Update harmony memory
            if new_fitness < np.max(self.fitness_scores):
                worst_index = np.argmax(self.fitness_scores)
                self.harmony_memory[worst_index] = new_harmony
                self.fitness_scores[worst_index] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.harmony_memory[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class HarrisHawk:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.rand(self.X.shape[1])
                A = np.random.rand()
                new_position = self.population[i] + A * (self.population[np.random.randint(self.pop_size)] - self.population[i]) + r

                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class HenryGasSolubility:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                A = np.random.rand()  # Random coefficient
                new_position = self.population[i] + A * (self.population[np.random.randint(self.pop_size)] - self.population[i])
                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class InvasiveWeed:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                new_position = self.population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features  # Return the indices of selected features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class KrillHerd:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                # Create a new position by perturbing the current position
                new_position = self.population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                # Update the position if the new fitness is better
                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class MothFlame:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                # Create a new position by perturbing the current position
                new_position = self.population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                # Update the position if the new fitness is better
                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class NonDominatedSortingGeneticOptimization:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            # Genetic operations: Selection, Crossover, Mutation (basic genetic operations)
            for i in range(self.pop_size):
                # Mimic genetic operations without full implementation
                new_position = self.population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                # Update the population if the new fitness is better
                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class NuclearReactionOptimization:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            # Nuclear reaction operations
            for i in range(self.pop_size):
                # Mimic nuclear reaction operations by adding noise to positions
                new_position = self.population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                new_fitness = self.fitness_function(new_position)

                # Update the population if the new fitness is better
                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class ParticleSwarm:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the population and velocities
        self.population = np.random.rand(pop_size, X.shape[1])
        self.velocities = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])
        self.personal_best = self.population.copy()
        self.personal_best_scores = self.fitness_scores.copy()

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] += r1 * (self.personal_best[i] - self.population[i]) + r2 * (self.population[np.argmin(self.fitness_scores)] - self.population[i])
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], 0, 1)

                # Update fitness score for the current particle
                self.fitness_scores[i] = self.fitness_function(self.population[i])

                # Update personal best if necessary
                if self.fitness_scores[i] < self.personal_best_scores[i]:
                    self.personal_best[i] = self.population[i]
                    self.personal_best_scores[i] = self.fitness_scores[i]

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class Pathfinder:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = None

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the population
        self.population = np.random.rand(pop_size, X.shape[1])
        self.fitness_scores = np.array([self.fitness_function(ind) for ind in self.population])

    def fitness_function(self, solution):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Penalize solutions with no selected features
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train[:, selected_features], self.y_train)
        y_pred = model.predict(self.X_test[:, selected_features])
        return -accuracy_score(self.y_test, y_pred)

    def search(self):
        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                new_position = self.population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)

                # Evaluate new position
                new_fitness = self.fitness_function(new_position)
                if new_fitness < self.fitness_scores[i]:
                    self.population[i] = new_position
                    self.fitness_scores[i] = new_fitness

            best_fitness = -np.min(self.fitness_scores)
            print(f"Generation {self.generation_counter}: Best fitness = {best_fitness}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(self.fitness_scores)  # Index of the best solution
        best_solution = self.population[best_solution_index]  # The solution itself (binary array)

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class QueuingSearch:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = []

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Return a high penalty for no features selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                new_position = population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                if self.fitness_function(new_position, X_train, X_test, y_train, y_test) < fitness_scores[i]:
                    population[i] = new_position

            print(f"Generation {self.generation_counter}: Best fitness = {-np.min(fitness_scores)}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class PlusLMinusR:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = []

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Return a high penalty for no features selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                L = np.mean(population, axis=0)
                R = np.random.rand(self.X.shape[1])
                new_position = L + R * (population[i] - L)
                new_position = np.clip(new_position, 0, 1)
                if self.fitness_function(new_position, X_train, X_test, y_train, y_test) < fitness_scores[i]:
                    population[i] = new_position

            print(f"Generation {self.generation_counter}: Best fitness = {-np.min(fitness_scores)}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class Sailfish:
    def __init__(self, X, y, pop_size=20, max_iter=10):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generation_counter = 0
        self.selected_features = []

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Return a high penalty for no features selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()

        for generation in range(self.max_iter):
            for i in range(self.pop_size):
                new_position = population[i] + np.random.normal(0, 0.1, size=self.X.shape[1])
                new_position = np.clip(new_position, 0, 1)
                if self.fitness_function(new_position, X_train, X_test, y_train, y_test) < fitness_scores[i]:
                    population[i] = new_position

            print(f"Generation {self.generation_counter}: Best fitness = {-np.min(fitness_scores)}")
            self.generation_counter += 1

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        best_solution_index = np.argmin(fitness_scores)  # Index of the best solution
        best_solution = population[best_solution_index]  # The solution itself (binary array)
        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]  # Indices of the selected features
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class ShuffleFrogLeaping:
    def __init__(self, X, y, pop_size=20, max_iter=10, leaping_rate=0.5):
        self.X = X  # Dataset features
        self.y = y  # Dataset labels
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.leaping_rate = leaping_rate
        self.selected_features = None

    def fitness_function(self, solution, X_train, X_test, y_train, y_test):
        selected_features = np.where(solution > 0.5)[0]
        if len(selected_features) == 0:
            return 1  # Return a high penalty for no features selected
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test[:, selected_features])
        return -accuracy_score(y_test, y_pred)

    def search(self):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Initialize population with random values
        population = np.random.rand(self.pop_size, self.X.shape[1])
        fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        start_time = time.time()
        for iteration in range(self.max_iter):
            # Sort indices based on fitness scores (lower is better)
            sorted_indices = np.argsort(fitness_scores)
            elite_frogs = population[sorted_indices[:self.pop_size // 2]]

            for i in range(self.pop_size // 2):
                if np.random.rand() < self.leaping_rate:
                    frog = elite_frogs[i]
                    new_frog = frog + np.random.uniform(-0.1, 0.1, size=frog.shape)
                    new_frog = np.clip(new_frog, 0, 1)
                    population[sorted_indices[i]] = new_frog

            fitness_scores = np.array([self.fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

        end_time = time.time()
        print(f"Total time taken for optimization: {end_time - start_time:.2f} seconds")

        # Find the index of the best solution
        best_solution_index = np.argmin(fitness_scores)
        best_solution = population[best_solution_index]

        # Extract the selected features from the best solution
        selected_indices = np.where(best_solution > 0.5)[0]
        selected_features = [features[i] for i in selected_indices]
        return selected_features

import os

import os
import pandas as pd

def feature_selection(X, y, full_data, MAX_ITER, output_dir):
    # Define paths for Data and Summary folders
    data_dir = os.path.join(output_dir, 'Data')
    summary_dir = os.path.join(output_dir, 'Summary')
    print("Welcome")

    # Ensure the output directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # List of feature selection algorithms to call
    algorithms = [
        CuckooSearch, EvolutionaryProgramming, Firefly, AdaptiveBacterialForaging,
        AntColony, ArtificialBeeColony, SineCosine, SocialSpider, Symbiotic,
        BacterialForaging, Bat, BigBangBigCrunch, Biogeography, TugOfWar,
        WaterCycle, WhaleOptimization, WhaleSwarmOptimization, CatSwarmOptimizer,
        ChickenSwarmOptimizer, ClonalSelectionOptimizer, CoralReefsOptimizer,
        FireworkOptimization, FlowerPollination, GravitationalSearch,
        GrayWolfOptimization, GreenHeronsOptimization, GreyWolfOptimizer,
        HarmonySearch, HarrisHawk, HenryGasSolubility, InvasiveWeed,
        KrillHerd, MothFlame, NonDominatedSortingGeneticOptimization,
        NuclearReactionOptimization, ParticleSwarm, Pathfinder,
        QueuingSearch, PlusLMinusR, Sailfish, ShuffleFrogLeaping
    ]

    # Iterate over each algorithm
    for algorithm in algorithms:
        algorithm_name = algorithm.__name__

        # Instantiate the algorithm class
        selector = algorithm(X, y, MAX_ITER)  # Assuming the class constructor takes X, y, and MAX_ITER

        # Call the search method to get selected features
        print(algorithm_name)
        selected_features = selector.search()

        # Generate summary CSV: Algorithm name, number of selected features, and feature list
        num_selected_features = len(selected_features)
        summary_data = {
            "Algorithm": [algorithm_name],
            "Number of Selected Features": [num_selected_features],
            "Selected Features": [", ".join(map(str, selected_features))]  # Ensure features are converted to string
        }
        summary_df = pd.DataFrame(summary_data)
        summary_file_path = os.path.join(summary_dir, f"{algorithm_name}_summary.csv")
        summary_df.to_csv(summary_file_path, index=False)

        # Generate selected data CSV: Full data with selected features and label
        selected_data_df = full_data[selected_features + ['label']].copy()
        selected_data_file_path = os.path.join(data_dir, f"{algorithm_name}_selected_data.csv")
        selected_data_df.to_csv(selected_data_file_path, index=False)

output_dir=''

X.shape

y.shape

data.shape

feature_selection(X, y, data, MAX_ITER,output_dir)

"""# GAN"""


