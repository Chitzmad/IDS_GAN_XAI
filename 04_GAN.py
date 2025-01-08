import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from datetime import datetime
 
start_time = datetime.now()
 
"""# GAN"""
 
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import RMSprop
 
 
# Function to create a basic GAN generator model
def create_standard_gan_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(output_dim, activation='tanh'))
    return model
 
def create_cgan_generator(latent_dim, output_dim, num_classes):
    # Define label input and embedding layer for labels
    label = Input(shape=(1,), name='label_input')
    label_embedding = Embedding(num_classes, latent_dim, input_length=1)(label)  # Embed to match `latent_dim`
    label_embedding = Flatten()(label_embedding)  # Flatten embedding to concatenate
 
    # Define noise input
    noise = Input(shape=(latent_dim,), name='noise_input')
 
    # Concatenate noise and label embedding
    combined_input = Concatenate()([noise, label_embedding])  # This shape is (latent_dim + latent_dim)
 
    # Build generator model with combined input
    x = Dense(256)(combined_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    generator_output = Dense(output_dim, activation='tanh')(x)
 
    # Create the model
    model = Model([noise, label], generator_output)
    return model
 
# Function to create a Wasserstein GAN (WGAN) generator model
def create_wgan_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(output_dim, activation='tanh'))
    return model
 
def generate_samples(generator, n_samples, latent_dim, gan_type, num_classes=None, cls=None):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
 
    if gan_type == "cGAN" and cls is not None:
        labels = np.full((n_samples, 1), cls)
        generated_samples = generator.predict([noise, labels])
    else:
        generated_samples = generator.predict(noise)
 
    return generated_samples
 
# def generate_data_with_gans(data, output_dir, base_name, latent_dim=100, samples_per_class=1000):
def generate_data_with_gans(data, output_dir, base_name, latent_dim, samples_per_class):
    os.makedirs(output_dir, exist_ok=True)
    classes = np.unique(data['label'])
    num_features = data.shape[1] - 1
    num_classes = len(classes)
 
    for gan_type in ["StandardGAN", "cGAN", "WGAN"]:
        all_generated_data = []
 
        for cls in classes:
            if gan_type == "StandardGAN":
                generator = create_standard_gan_generator(latent_dim, num_features)
                generated_samples = generate_samples(generator, samples_per_class, latent_dim, gan_type)
 
            elif gan_type == "cGAN":
                generator = create_cgan_generator(latent_dim, num_features, num_classes)
                generated_samples = generate_samples(generator, samples_per_class, latent_dim, gan_type, num_classes, cls)
 
            elif gan_type == "WGAN":
                generator = create_wgan_generator(latent_dim, num_features)
                generated_samples = generate_samples(generator, samples_per_class, latent_dim, gan_type)
 
            generated_label = np.full((samples_per_class, 1), cls)
            generated_data = np.hstack((generated_samples, generated_label))
            all_generated_data.append(generated_data)
 
        all_generated_data = np.vstack(all_generated_data)
        df_generated = pd.DataFrame(all_generated_data, columns=[*data.columns[:-1], 'label'])
        df_generated = df_generated.round(3)
 
        filename = os.path.join(output_dir, f"{base_name}_{gan_type}.csv")
        df_generated.to_csv(filename, index=False)
        print(f"Data for {gan_type} generated and saved successfully as:", filename)
 
import os
import pandas as pd
 
# Define the input and output directories
#input_dir = './data_extract'
input_dir = './data_extract_individual'
#output_dir = './GAN_individual'
output_dir = './GAN_individual'
latent_dim = 134
 
# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('_selected_data.csv'):  # Process only files with the specific suffix
        file_path = os.path.join(input_dir, filename)
 
        # Extract the base file name (remove "_selected_data.csv")
        base_name = filename.replace('_selected_data.csv', '')
 
        # Read the data from the CSV file
        data = pd.read_csv(file_path)
 
        # Read the CSV file, ignoring the header row (index_col is set to None to avoid using any column as index)
        df1 = pd.read_csv(file_path, skiprows=[0])
 
        # Get the last column name
        last_column = df1.columns[-1]

        # Count the frequency of each label in the last column
        label_counts = df1[last_column].value_counts()
 
        # # Compute the median of these label frequencies
        # median_value = int(label_counts.median())
 
        # print("Median value of label frequencies:", median_value) 
        # Generate data with GANs for each file
        print(f"Processing file: {base_name}")
        median_value=15000
        generate_data_with_gans(data, output_dir, base_name, latent_dim, median_value)
        print(f"Finished processing file: {base_name}\n")
        
        
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
