import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Initialize dataset parameters
#Change the filename.

input_directory = 'GAN'  # Update with the directory containing input CSV files
output_directory = 'Graphs'  # Update with the directory for results

dataset_percent = 50
# input_file = 'extracted_merged_dataset_stratified_10.csv'

# print(type(X), type(y))

import os
# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)
 
from datetime import timedelta
import logging
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
#  MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef
from sklearn.metrics import jaccard_score, cohen_kappa_score, hamming_loss, zero_one_loss, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, GRU, SimpleRNN
from keras.utils import to_categorical
import traceback
import csv
import warnings
from sklearn.semi_supervised import SelfTrainingClassifier
from collections import defaultdict
 
warnings.filterwarnings("ignore")
 
k_fold = 5
# dataset_percent = 100
 
 
# Initialize CSV file and columns
# output_file = input_file.replace('.csv', '_graph_results.csv')
csv_columns = ['Algorithm', 'Fold', 'Train Time (s)', 'Test Time (s)', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 
               'Fbeta Score', 'Matthews Correlation Coefficient', 'Jaccard Score', 'Cohen Kappa Score', 
               'Hamming Loss', 'Zero One Loss', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error',
               'Balanced Accuracy', 'R2 Score']
 
# Function to handle metric calculation
def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    metrics['Fbeta Score'] = fbeta_score(y_true, y_pred, beta=1.0, average='weighted', zero_division=1)
    metrics['Matthews Correlation Coefficient'] = matthews_corrcoef(y_true, y_pred)
    metrics['Jaccard Score'] = jaccard_score(y_true, y_pred, average='weighted', zero_division=1)
    metrics['Cohen Kappa Score'] = cohen_kappa_score(y_true, y_pred)
    metrics['Hamming Loss'] = hamming_loss(y_true, y_pred)
    metrics['Zero One Loss'] = zero_one_loss(y_true, y_pred)
    metrics['Mean Absolute Error'] = mean_absolute_error(y_true, y_pred)
    metrics['Mean Squared Error'] = mean_squared_error(y_true, y_pred)
    metrics['Root Mean Squared Error'] = np.sqrt(metrics['Mean Squared Error'])
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['R2 Score'] = r2_score(y_true, y_pred)
    
    return metrics

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Create directories if they don't exist
os.makedirs('confusion_matrices', exist_ok=True)
os.makedirs('ROC_curves', exist_ok=True)
os.makedirs('PR_curves', exist_ok=True)

def append_confusion_matrix(algo_name, y_true, y_pred, fold,file_name):
    file_name.replace('.csv','')
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]  # Get the number of classes

    # Generate class labels for rows and columns
    actual_labels = [f'actual_class_{i}' for i in range(num_classes)]
    predicted_labels = [f'predicted_class_{i}' for i in range(num_classes)]
    
    with open('confusion_matrix.csv', 'a', newline='') as cm_file:
        writer = csv.writer(cm_file)

        # Write algorithm name
        writer.writerow([f'Dataset: {file_name}'])
        writer.writerow([f'Algorithm: {algo_name}'])
        writer.writerow([f'Fold: {fold}'])
        
        # Write header row with predicted class labels
        writer.writerow([''] + predicted_labels)

        # Write confusion matrix with actual class labels
        for i, row in enumerate(cm):
            writer.writerow([actual_labels[i]] + row.tolist())

        # Add 5 blank lines to separate from the next confusion matrix
        writer.writerow([])
        writer.writerow([])
        writer.writerow([])
        writer.writerow([])
        writer.writerow([])

    # Plot and save confusion matrix as PNG file
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {algo_name} (Fold {fold})')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, predicted_labels, rotation=45)
    plt.yticks(tick_marks, actual_labels)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrices/{file_name}_confusion_matrix_{algo_name}fold{fold}.png')
    # plt.show()
    plt.close()

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Binarize the output for multi-class ROC calculation
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_bin = label_binarize(y_pred, classes=np.unique(y_true))  # Ensure y_pred is binarized

    if num_classes == 2:
        fpr[1], tpr[1], _ = roc_curve(y_true_bin[:, 0], y_pred)
        roc_auc[1] = auc(fpr[1], tpr[1])
        
        # Plot and save ROC curve as PNG file
        plt.figure()
        plt.plot(fpr[1], tpr[1], lw=2,
                 label=f'ROC curve (area = {roc_auc[1]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic: {algo_name} (Fold {fold})')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC_curves/{file_name}_roc_curve_{algo_name}fold{fold}.png')
        # plt.show()
        plt.close()
        
    else:
        # Plot ROC curves for each class in the same graph
        plt.figure()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic: {algo_name} (Fold {fold})')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC_curves/{file_name}_roc_curve_{algo_name}fold{fold}.png')
        # plt.show()
        plt.close()

    # Calculate Precision-Recall curve and AUC for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()

    if num_classes == 2:
        precision[1], recall[1], _ = precision_recall_curve(y_true_bin[:, 0], y_pred)
        pr_auc[1] = auc(recall[1], precision[1])

        # Plot and save Precision-Recall curve as PNG file
        plt.figure()
        plt.plot(recall[1], precision[1], lw=2,
                 label=f'Precision-Recall curve (area = {pr_auc[1]:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {algo_name} (Fold {fold})')
        plt.legend(loc="lower left")
        plt.savefig(f'PR_curves/{file_name}_precision_recall_curve_{algo_name}fold{fold}.png')
        # plt.show()
        plt.close()

    else:
        # Plot Precision-Recall curves for each class in the same graph
         plt.figure()
        
         # Plot Precision-Recall curves for each class in the same graph
         for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
            plt.plot(recall[i], precision[i], lw=2,
                     label=f'Precision-Recall curve of class {i} (area = {pr_auc[i]:.2f})')

         # Plot settings
         plt.xlim([0.0, 1.0])
         plt.ylim([0.0, 1.05])
         plt.xlabel('Recall')
         plt.ylabel('Precision')
         plt.title(f'Precision-Recall Curve: {algo_name} (Fold {fold})')
         plt.legend(loc="lower left")
         plt.savefig(f'PR_curves/{file_name}_precision_recall_curve_{algo_name}fold{fold}.png')
        #  plt.show()
         plt.close()

# Example usage:
# append_confusion_matrix("Decision Tree", y_test, y_pred, fold=1)


import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Create directories if they don't exist
# os.makedirs('confusion_matrices', exist_ok=True)
# os.makedirs('ROC_curves', exist_ok=True)
# os.makedirs('PR_curves', exist_ok=True)


# Modified run_algorithm function to include confusion matrix logging
def run_algorithm(algo_name, model, X_train, y_train, X_test, y_test, fold,file_name):
    try:
        file_name.replace('.csv','')
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
 
        start_test = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - start_test

        # Compute and log confusion matrix
        append_confusion_matrix(algo_name, y_test, y_pred,fold,file_name)
 
        # Compute metrics
        if algo_name == 'ElasticNet':  # Handle ElasticNet as a regression model
            metrics = {}
            metrics['Mean Absolute Error'] = mean_absolute_error(y_test, y_pred)
            metrics['Mean Squared Error'] = mean_squared_error(y_test, y_pred)
            metrics['Root Mean Squared Error'] = np.sqrt(metrics['Mean Squared Error'])
            metrics['R2 Score'] = r2_score(y_test, y_pred)
        else:
            # Compute classification metrics
            metrics = compute_metrics(y_test, y_pred)
        metrics.update({'Train Time (s)': train_time, 'Test Time (s)': test_time})
        
        # Log results to CSV
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([algo_name, fold] + [metrics.get(m, -1) for m in csv_columns[2:]])
 
        print(f"{algo_name} | Fold: {fold} | Train Time: {train_time:.2f}s | Test Time: {test_time:.2f}s")
        
    except Exception as e:
        # Log error case
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([algo_name, fold] + [-1 for _ in csv_columns[2:]])
        print(f"Error in {algo_name}: {traceback.format_exc()}")

 

 
# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
 
# List of algorithms
algorithms = {


            'Naive Bayes': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis(),
            'SVM': SVC(kernel='linear', max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'SGD Classifier': SGDClassifier(),
            'KNN': KNeighborsClassifier(),
            'ElasticNet': ElasticNet(),
            'Perceptron': Perceptron(),
            'Logistic Regression': LogisticRegression(),
            'Bagging': BaggingClassifier(),
            'K-Means': KMeans(n_clusters=3),
            'Nearest Centroid Classifier': NearestCentroid(),
            'XGBoost': XGBClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            ########'RNN': create_rnn((28, 28)),
            'RBM + Logistic Regression': Pipeline(steps=[('rbm', BernoulliRBM(n_components=100, learning_rate=0.06, n_iter=10, random_state=42)),('logistic', LogisticRegression())]),
            #'Voting Classifier': VotingClassifier(estimators=[('lr', LogisticRegression()),('rf', RandomForestClassifier()),('gnb', GaussianNB())], voting='hard'),
            'Random Forest': RandomForestClassifier(n_estimators=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=10),
            #'Stacking Classifier': StackingClassifier(estimators=[('log_reg', LogisticRegression()),('knn', KNeighborsClassifier(n_neighbors=3))],final_estimator=LogisticRegression(),n_jobs=-1),
            'MLP Classifier': MLPClassifier(),
            ######### 'GRU': create_gru((28, 28)),
            ######### 'LSTM': create_lstm((28, 28)),
            ######### 'CNN': create_cnn((28, 28, 1)),
            ######### 'Autoencoder': create_autoencoder((28,)),
            #'LightGBM': LGBMClassifier(),
            #'CatBoost': CatBoostClassifier(),
            #'Self-Training': SelfTrainingClassifier(LogisticRegression()),
            'Isolation Forest': IsolationForest(),
            # 'One-Class SVM': OneClassSVM(kernel='linear', max_iter=1000)
            # 'Deep Belief Network': "Implement DBN",  # Placeholder for DBN
            # 'Restricted Boltzmann Machine': "Implement RBM",  # Placeholder for RBM
            # 'Genetic Algorithm': ga.GeneticAlgorithm(),  # Placeholder for Genetic Algorithm-based 
            # 'Bayesian Network': BayesianNetwork([('A', 'B'), ('B', 'C')]),  # Example Bayesian Network
            # 'Fuzzy Logic': "Implement Fuzzy Logic",  # Placeholder for Fuzzy Logic systems
            # 'Conditional Random Field (CRF)': "Implement CRF",  # Placeholder for CRF
                }

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler for logging to a file in append mode
file_handler = logging.FileHandler('execution_log.log', mode='a')
file_handler.setLevel(logging.DEBUG)

# Create a console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define a formatter and attach it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Running algorithms in k-fold
start_time = time.time()
logger.info('Starting k-fold cross-validation')

# Process all CSV files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.csv'):  # Ensure only CSV files are processed
        input_file = os.path.join(input_directory, file_name)
        output_file = os.path.join(output_directory, input_file.replace('.csv', '_graph_results.csv'))
        
        print(f"Processing file: {input_file}")
        
        try:
            # Load dataset
            df = pd.read_csv(input_file)
            # X = df.iloc[:, :-1]
            X = pd.get_dummies(df.iloc[:, :-1])
            y = df.iloc[:, -1]
            
            if dataset_percent < 100:
                _, X, _, y = train_test_split(X, y, test_size=dataset_percent / 100, stratify=y)

            # Encode categorical features
            label_encoder = LabelEncoder()
            for column in X.columns:
                if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                    X[column] = label_encoder.fit_transform(X[column])
            y = LabelEncoder().fit_transform(y)
            
            # # Apply iterative imputation to handle missing data
            imputer = IterativeImputer()
            X = imputer.fit_transform(X)
            
            # Prepare output CSV file
            output_file = os.path.join(output_directory, file_name.replace('.csv', '_graph_results.csv'))
            if not os.path.exists(output_file):
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_columns)
            
            # K-Fold Cross Validation
            kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
            
            # Running algorithms in k-fold
            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                for algo_name, model in algorithms.items():
                    logger.info(f'Running algorithm: {algo_name}')
                    run_algorithm(algo_name, model, X_train, y_train, X_test, y_test, fold,file_name)
                    logger.info(f'Algorithm {algo_name} completed')
            
            print(f"File {input_file} processed successfully.")
        
        except Exception as e:
            print(f"Error processing file {input_file}: {traceback.format_exc()}")




end_time = time.time()
execution_time = end_time - start_time

# Calculate the execution time in days, hours, minutes, and seconds
execution_delta = timedelta(seconds=execution_time)
days = execution_delta.days
hours = execution_delta.seconds // 3600
minutes = (execution_delta.seconds // 60) % 60
seconds = execution_delta.seconds % 60

# Log the total execution time
logger.info(f'All algorithms have been executed. Results are saved in {output_file}')
logger.info(f'Execution Time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds')
print(f"Execution Time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

# Load the CSV file and sort the results by F1 Score
df = pd.read_csv(output_file)
df_sorted = df.sort_values(by='F1 Score', ascending=False)

# Save the sorted results back to the CSV file
df_sorted.to_csv(output_file, index=False)

# Print the first few rows to verify
print(df_sorted.head())

# Append execution time and input file name to a txt file
with open('execution_times.txt', 'a') as log_file:
    log_file.write(f"Input file: {input_file} | Execution Time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds\n")
logger.info(f'Execution time logged in execution_times.txt')






