import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the data from a CSV file
# Replace 'your_data.csv' with your actual CSV file path.
data = pd.read_csv('your_data.csv')

# Step 2: Preprocess the data
# Assuming the last column is the target variable and the rest are features.
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Calculate mean, variance and prior for each class
def calculate_stats(X_train, y_train):
    stats = {}
    for class_value in np.unique(y_train):
        X_class = X_train[y_train == class_value]
        stats[class_value] = {
            "mean": X_class.mean(),
            "var": X_class.var(),
            "prior": len(X_class) / len(X_train)
        }
    return stats

# Step 5: Calculate the probability density function for Gaussian distribution
def gaussian_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2 / (2 * var)))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

# Step 6: Classify test points based on Naive Bayes principles
def classify(stats, X_test):
    predictions = []
    for i in range(len(X_test)):
        class_probs = {}
        for class_value, class_stats in stats.items():
            # Start with the prior probability
            class_probs[class_value] = np.log(class_stats["prior"])
            for j in range(len(X_test.columns)):
                mean = class_stats["mean"].iloc[j]
                var = class_stats["var"].iloc[j]
                x = X_test.iloc[i, j]
                class_probs[class_value] += np.log(gaussian_probability(x, mean, var))
        # Choose the class with the highest probability
        best_class = max(class_probs, key=class_probs.get)
        predictions.append(best_class)
    return predictions

# Step 7: Train the model by calculating the stats
stats = calculate_stats(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = classify(stats, X_test)

# Step 9: Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the custom Naive Bayes classifier: {accuracy * 100:.2f}%")
