import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
num_classes = len(np.unique(y))

# Compute the mean of each feature for each class
class_means = np.array([X[y == c].mean(axis=0) for c in range(num_classes)])

# Generate soft targets
soft_targets = []
for x in X:
    # Compute absolute distances from class means for each feature
    distances = np.abs(class_means - x)
    
    # Compute reciprocals of distances (add small epsilon to avoid divide-by-zero)
    reciprocals = 1 / (distances + 1e-8)
    
    # Softmax for each feature
    feature_probs = np.exp(reciprocals) / np.sum(np.exp(reciprocals), axis=0)
    
    # Average across features
    avg_probs = feature_probs.mean(axis=1)
    
    # Normalize to create final soft target probabilities
    final_probs = np.exp(avg_probs) / np.sum(np.exp(avg_probs))
    soft_targets.append(final_probs)

soft_targets = np.array(soft_targets)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, soft_targets, test_size=0.2, random_state=42)

# Save to CSV files
pd.DataFrame(X_train).to_csv("iris_features_train.csv", index=False, header=False)
pd.DataFrame(y_train).to_csv("iris_soft_targets_train.csv", index=False, header=False)
pd.DataFrame(X_test).to_csv("iris_features_test.csv", index=False, header=False)
pd.DataFrame(y_test).to_csv("iris_soft_targets_test.csv", index=False, header=False)

print("CSVs created: 'iris_features_train.csv', 'iris_soft_targets_train.csv', 'iris_features_test.csv', 'iris_soft_targets_test.csv'")

