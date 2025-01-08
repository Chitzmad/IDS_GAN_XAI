import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt


# Define the output folder
output_folder = 'shap_lime_output_1'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

dataset_percent = 50

# Load the dataset with only the selected common feature columns
df = pd.read_csv(r'ParticleSwarm_WhaleSwarmOptimization_32_selected_data.csv')
X = df.drop('label', axis=1)
y = df['label']

# Get unique class names from the dataset
class_names = y.unique()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Save SHAP summary plot to the output folder
shap_plot_path = os.path.join(output_folder, 'shap_summary_plot.png')
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.savefig(shap_plot_path)
plt.close()

# Create LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, 
    feature_names=X.columns, 
    class_names=class_names,  # Dynamically set class names
    mode='classification'
)

# Explain a single instance using LIME
lime_exp = lime_explainer.explain_instance(X_test.values[0], model.predict_proba, num_features=10)

# Save LIME explanation as a visual plot
lime_plot_path = os.path.join(output_folder, 'lime_explanation.png')
fig = lime_exp.as_pyplot_figure()
fig.savefig(lime_plot_path)
plt.close(fig)

print(f"SHAP summary plot saved to: {shap_plot_path}")
print(f"LIME explanation saved to: {lime_plot_path}")
