import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

param_grid = {
    "n_neighbors": [3,5,7,9,11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_knn = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


y_pred = best_knn.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\n Validation Accuracy (KNN): {acc * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_val, y_pred))
print("\n Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
