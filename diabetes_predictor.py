#importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#loading dataset
diabetes = pd.read_csv("diabetes.csv")

#test
print(diabetes.info())
print(diabetes.head())

#exploratory analysis of outcome
sns.countplot(x = "Outcome", data = diabetes)
plt.show()


X = diabetes.drop("Outcome", axis = 1)
y = diabetes["Outcome"]

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 66)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#
# K-Nearest Neighbors (KNN)
print("K-Nearest Neighbors")
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train_scaled, y_train)
    training_accuracy.append(knn.score(X_train_scaled, y_train))
    test_accuracy.append(knn.score(X_test_scaled, y_test))

plt.plot(neighbors_settings, training_accuracy, label = "Training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.title("KNN Accuracy vs. Number of Neighbors")
plt.show()

#Best KNN model
best_knn = KNeighborsClassifier(n_neighbors = 9)
best_knn.fit(X_train_scaled, y_train)
print("KNN Accuracy on training set: {:.2f}".format(best_knn.score(X_train_scaled, y_train)))
print("KNN Accuracy on test set: {:.2f}".format(best_knn.score(X_test_scaled, y_test)))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Decision Tree
print("\nDecision Tree")
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Decision Tree Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Decision Tree Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# Pruned Decision Tree
pruned_tree = DecisionTreeClassifier(max_depth = 3, random_state = 0)
pruned_tree.fit(X_train, y_train)
print("Pruned Decision Tree Accuracy on training set: {:.3f}".format(pruned_tree.score(X_train, y_train)))
print("Pruned Decision Tree Accuracy on test set: {:.3f}".format(pruned_tree.score(X_test, y_test)))

# Feature importance for Decision Tree
def plot_feature_importances(model, feature_names):
    plt.figure(figsize = (8,6))
    n_features = len(feature_names)
    plt.barh(range(n_features), model.feature_importances_, align = "center")
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title("Feature Importance")
    plt.show()

plot_feature_importances(pruned_tree, X.columns)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Multi-Layer Perceptron (MLP)
print("\nMulti-Layer Perceptron")
#mlp = MLPClassifier(random_state = 42)
#mlp = MLPClassifier(hidden_layer_sizes = (25, ), alpha = 0.01, activation = 'tanh', learning_rate = 'adaptive', learning_rate_init = 0.01, max_iter = 500, tol = 1e-4, solver = 'sgd', batch_size = 32,  random_state = 42)
#mlp.fit(X_train_scaled, y_train)
#print("MLP Accuracy on training set: {:.4f}".format(mlp.score(X_train_scaled, y_train)))
#print("MLP Accuracy on test set: {:.4f}".format(mlp.score(X_test_scaled, y_test)))
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

#using feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


# Visualization of MLP weights
plt.figure(figsize = (20, 5))
plt.imshow(mlp.coefs_[0], interpolation = "none", cmap = "viridis")
plt.yticks(range(X.shape[1]), X.columns)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.title("MLP Weight Matrix")
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Confusion Matrix and Classification Report
y_pred = best_knn.predict(X_test_scaled)
print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report (KNN):\n", classification_report(y_test, y_pred))

y_pred_tree = pruned_tree.predict(X_test)
print("\nConfusion Matrix (Decision Tree):\n", confusion_matrix(y_test, y_pred_tree))
print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred_tree))

y_pred_mlp = mlp.predict(X_test_scaled)
print("\nConfusion Matrix (MLP):\n", confusion_matrix(y_test, y_pred_mlp))
print("\nClassification Report (MLP):\n", classification_report(y_test, y_pred_mlp))
