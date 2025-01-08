import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

input_directory = './GAN'  # Update with the directory containing input CSV files
output_directory = 'Classwise_Results'  # Update with the directory for results

#print(type(X), type(y))  # Debugging: Ensure they are either Pandas or NumPy

import os
# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, GRU, SimpleRNN
from keras.utils import to_categorical
import traceback
import csv
import warnings
from collections import defaultdict
from sklearn.semi_supervised import SelfTrainingClassifier


warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Dataset parameters
k_fold = 5  # Number of folds for cross-validation
dataset_percent = 50  # Percentage of the dataset to use


# CSV columns
csv_columns = [
    'Algorithm', 'Fold', 'Train Time (s)', 'Test Time (s)', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
    'Fbeta Score', 'Matthews Correlation Coefficient', 'Jaccard Score', 'Cohen Kappa Score',
    'Hamming Loss', 'Zero One Loss', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error',
    'Balanced Accuracy', 'R2 Score'
]
class_metrics_columns = [
    'Algorithm', 'Fold', 'Class', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
    'Fbeta Score', 'Matthews Correlation Coefficient', 'Jaccard Score', 'Cohen Kappa Score',
    'Hamming Loss', 'Zero One Loss', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error',
    'Balanced Accuracy', 'R2 Score'
]

# Function to compute class-wise metrics
def compute_classwise_metrics(y_true, y_pred):
    class_metrics = defaultdict(dict)
    classes = np.unique(y_true)
    for class_index in classes:
        # true_class_name = class_names[class_index]
        true_class_name = class_index
        y_true_class = (y_true == class_index).astype(int)
        y_pred_class = (y_pred == class_index).astype(int)

        class_metrics[true_class_name] = {
            'Accuracy': round(accuracy_score(y_true_class, y_pred_class), 3),
            'Precision': round(precision_score(y_true, y_pred, labels=[class_index], average='weighted', zero_division=1), 3),
            'Recall': round(recall_score(y_true, y_pred, labels=[class_index], average='weighted', zero_division=1), 3),
            'F1 Score': round(f1_score(y_true, y_pred, labels=[class_index], average='weighted', zero_division=1), 3),
            'Fbeta Score': round(fbeta_score(y_true, y_pred, labels=[class_index], beta=1.0, average='weighted', zero_division=1), 3),
            'Matthews Correlation Coefficient': round(matthews_corrcoef(y_true_class, y_pred_class), 3),
            'Jaccard Score': round(jaccard_score(y_true, y_pred, labels=[class_index], average='weighted', zero_division=1), 3),
            'Cohen Kappa Score': round(cohen_kappa_score(y_true_class, y_pred_class), 3),
            'Hamming Loss': round(hamming_loss(y_true_class, y_pred_class), 3),
            'Zero One Loss': round(zero_one_loss(y_true_class, y_pred_class), 3),
            'Mean Absolute Error': round(mean_absolute_error(y_true_class, y_pred_class), 3),
            'Mean Squared Error': round(mean_squared_error(y_true_class, y_pred_class), 3),
            'Root Mean Squared Error': round(np.sqrt(mean_squared_error(y_true_class, y_pred_class)), 3),
            'Balanced Accuracy': round(balanced_accuracy_score(y_true_class, y_pred_class), 3),
            'R2 Score': round(r2_score(y_true_class, y_pred_class), 3),
        }
    return class_metrics

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=1),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=1),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=1),
        'Fbeta Score': fbeta_score(y_true, y_pred, beta=1.0, average='weighted', zero_division=1),
        'Matthews Correlation Coefficient': matthews_corrcoef(y_true, y_pred),
        'Jaccard Score': jaccard_score(y_true, y_pred, average='weighted', zero_division=1),
        'Cohen Kappa Score': cohen_kappa_score(y_true, y_pred),
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Zero One Loss': zero_one_loss(y_true, y_pred),
        'Mean Absolute Error': mean_absolute_error(y_true, y_pred),
        'Mean Squared Error': mean_squared_error(y_true, y_pred),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred),
    }

# Function to run and log algorithm results
def run_algorithm(algo_name, model, X_train, y_train, X_test, y_test, fold, output_file, class_metrics_file):
    try:
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_test = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - start_test

        metrics = compute_metrics(y_test, y_pred)
        metrics.update({'Train Time (s)': train_time, 'Test Time (s)': test_time})
        class_metrics = compute_classwise_metrics(y_test, y_pred)

        # Log results
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([algo_name, fold] + [metrics.get(m, -1) for m in csv_columns[2:]])

        with open(class_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for class_label, cm in class_metrics.items():
                writer.writerow([algo_name, fold, class_label] + [cm.get(m, -1) for m in class_metrics_columns[3:]])

        print(f"{algo_name} | Fold: {fold} | Train Time: {train_time:.2f}s | Test Time: {test_time:.2f}s")
    except Exception as e:
        print(f"Error in {algo_name}: {traceback.format_exc()}")

# Loop through input files
for input_file in os.listdir(input_directory):
    if input_file.endswith('.csv'):
        input_path = os.path.join(input_directory, input_file)
        output_file = os.path.join(output_directory, input_file.replace('.csv', '_results.csv'))
        class_metrics_file = os.path.join(output_directory, input_file.replace('.csv', '_class_results.csv'))

        df = pd.read_csv(input_path)
        X = pd.get_dummies(df.iloc[:, :-1])
        y = df.iloc[:, -1] 
        class_names = y.unique()
        # y = LabelEncoder().fit_transform(df.iloc[:, -1])
        # class_names = LabelEncoder().fit(df.iloc[:, -1]).classes_
        # X = MinMaxScaler().fit_transform(X)

        if dataset_percent < 100:
            _, X, _, y = train_test_split(X, y, test_size=dataset_percent / 100, stratify=y)

        with open(output_file, 'w', newline='') as f:
            csv.writer(f).writerow(csv_columns)
        with open(class_metrics_file, 'w', newline='') as f:
            csv.writer(f).writerow(class_metrics_columns)

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)


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

        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {len(y)}")      

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            # X_train, X_test = X[train_idx], X[test_idx]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for algo_name, model in algorithms.items():
                run_algorithm(algo_name, model, X_train, y_train, X_test, y_test, fold, output_file, class_metrics_file)