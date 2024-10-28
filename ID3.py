import math
import pandas as pd

# Calculate the entropy
def entropy(data):
    labels = data.iloc[:, -1]
    total = len(labels)
    counts = labels.value_counts()
    entropy_value = 0
    for count in counts:
        prob = count / total
        entropy_value -= prob * math.log2(prob)
    return entropy_value

# Calculate information gain
def info_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

# Build the ID3 decision tree
def id3(data):
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:  # If all labels are the same
        return labels.iloc[0]
    
    best_feature = max(data.columns[:-1], key=lambda f: info_gain(data, f))
    tree = {best_feature: {}}
    
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value].drop(columns=[best_feature])
        tree[best_feature][value] = id3(subset)
    
    return tree

# Predict for a new instance
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = instance[feature]
    return predict(tree[feature][value], instance)

# Load dataset from CSV
data = pd.read_csv('ID3.csv')

# Build the decision tree
tree = id3(data)
print("Decision Tree:", tree)

# Predict for a new sample
new_sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
prediction = predict(tree, new_sample)
print("Prediction for new sample:", prediction)
