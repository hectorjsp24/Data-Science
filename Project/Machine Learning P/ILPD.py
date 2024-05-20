from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the dataset
file_path = 'Indian Liver Patient Dataset (ILPD).csv'
columns = ["Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Alkphos",
           "Sgpt", "Sgot", "Total Proteins", "Albumin", "A/G Ratio", "Liver Cancer"]
# Load excel file
data = pd.read_csv(file_path, names=columns)

# Assuming gender is listed as 'Male'/'Female'
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Drop rows with missing values
data = data.dropna()

# 1. Target attribute
target_attribute = 'Liver Cancer'

# 2. One instance of interest
instance_index = np.random.randint(len(data))
instance_of_interest = data.loc[instance_index]

# 3. Attribute of interest
attribute_of_interest = 'Alkphos'

# 4. Subset of rows of interest
gender_value = 1  # Male
subset = data[data['Gender'] == gender_value]

# 5. Cost matrix (benefits)
# Define the benefit matrix such that true positive has the highest benefit,
# followed by true negative, and false positives/negatives have lower benefits.
benefit_matrix = np.array([[10, 0], [0, 5]])

# 1. Histograms (or value frequencies)
# Show histograms for the target variable and selected other variables
selected_variables = ["Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Alkphos",
                      "Sgpt", "Sgot", "Total Proteins", "Albumin", "A/G Ratio"]
num_cols = 2
num_rows = (len(selected_variables) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 10))

for idx, variable in enumerate(selected_variables):
    row = idx // num_cols
    col = idx % num_cols
    if variable != "Liver Cancer":
        sns.histplot(data[variable], bins=20, kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Histogram of {variable}')
        axes[row, col].set_xlabel(variable)
        axes[row, col].set_ylabel('Frequency')
    else:
        value_counts = data[variable].value_counts()
        axes[row, col].bar(value_counts.index, value_counts.values)
        axes[row, col].set_title(f'Value frequencies for {variable}')
        axes[row, col].set_xlabel(variable)
        axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create subplots for scatterplots
num_subplots = len(columns) - 1  # Excluding the target variable
num_rows = (num_subplots + 1) // 2
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))

# Show scatterplots showing the relationship between each predictor and the target variable
predictors = ["Age", "Total Bilirubin", "Direct Bilirubin", "Alkphos",
              "Sgpt", "Sgot", "Total Proteins", "Albumin", "A/G Ratio"]

for i, predictor in enumerate(predictors):
    row = i // num_cols
    col = i % num_cols
    sns.scatterplot(data=data, x="Liver Cancer",
                    y=predictor, ax=axes[row, col])
    axes[row, col].set_title(f'Scatterplot of Liver Cancer vs {predictor}')
    axes[row, col].set_xlabel('Liver Cancer')
    axes[row, col].set_ylabel(predictor)

# Adjust layout
plt.tight_layout()
plt.show()

# Define selected features
selected_features = ['Age', 'Alkphos', 'Total Bilirubin', 'Sgpt']

# Define predictors and target variable for supervised models
X = data[selected_features]
y = data['Liver Cancer']


# Split the data into a training set and a test set (80% train, 20% test) for supervised models
# 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature Rescaling or Normalization
scaler = StandardScaler()
scaled_subset_data = scaler.fit_transform(X)

# Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluation
dt_predictions = dt_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)

# Dictionary to store performance metrics for each model
performance_metrics = {}

# Decision Tree classifier evaluation
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_f1_score = f1_score(y_test, dt_predictions)
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
performance_metrics['Decision_Tree'] = {
    'Accuracy': dt_accuracy, 'F1 Score': dt_f1_score}

# Random Forest classifier evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_f1_score = f1_score(y_test, rf_predictions)
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
performance_metrics['Random_Forest'] = {
    'Accuracy': rf_accuracy, 'F1 Score': rf_f1_score}

# Print confusion matrix for Decision Tree classifier
print("Confusion Matrix for Decision Tree:")
print(dt_conf_matrix)

# Print confusion matrix for Random Forest classifier
print("\nConfusion Matrix for Random Forest:")
print(rf_conf_matrix)

# K-Means clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(scaled_subset_data)
kmeans_silhouette = silhouette_score(scaled_subset_data, kmeans.labels_)
kmeans_davies_bouldin = davies_bouldin_score(
    scaled_subset_data, kmeans.labels_)
performance_metrics['KMeans'] = {
    'Silhouette Score': kmeans_silhouette, 'Davies-Bouldin Score': kmeans_davies_bouldin}

# Hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(scaled_subset_data)
agg_silhouette = silhouette_score(scaled_subset_data, agg_clustering.labels_)
agg_davies_bouldin = davies_bouldin_score(
    scaled_subset_data, agg_clustering.labels_)
performance_metrics['Hierarchical_Clustering'] = {
    'Silhouette Score': agg_silhouette, 'Davies-Bouldin Score': agg_davies_bouldin}

# Evaluate K-Means clustering
kmeans_silhouette = silhouette_score(scaled_subset_data, kmeans.labels_)
kmeans_davies_bouldin = davies_bouldin_score(
    scaled_subset_data, kmeans.labels_)

# Evaluate Hierarchical clustering
agg_silhouette = silhouette_score(scaled_subset_data, agg_clustering.labels_)
agg_davies_bouldin = davies_bouldin_score(
    scaled_subset_data, agg_clustering.labels_)

# Plotting
plt.figure(figsize=(10, 8))

# Display performance metrics for each model
for model, metrics in performance_metrics.items():
    print(f"Performance Metrics for {model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")

# Decision Tree Feature Importance
dt_feature_importance = dt_classifier.feature_importances_
dt_feature_importance_df = pd.DataFrame(
    {'Feature': X.columns, 'Importance': dt_feature_importance})
dt_feature_importance_df = dt_feature_importance_df.sort_values(
    by='Importance', ascending=False)

print("Decision Tree Feature Importance:")
print(dt_feature_importance_df)

# Random Forest Feature Importance
rf_feature_importance = rf_classifier.feature_importances_
rf_feature_importance_df = pd.DataFrame(
    {'Feature': X.columns, 'Importance': rf_feature_importance})
rf_feature_importance_df = rf_feature_importance_df.sort_values(
    by='Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_feature_importance_df)

# Plot clusters for K-Means
plt.subplot(2, 1, 1)
plt.scatter(scaled_subset_data[:, 0], scaled_subset_data[:, 1],
            c=kmeans.labels_, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot clusters for Hierarchical Clustering
plt.subplot(2, 1, 2)
plt.scatter(scaled_subset_data[:, 0], scaled_subset_data[:, 1],
            c=agg_clustering.labels_, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
