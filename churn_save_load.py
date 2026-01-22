import joblib
import pandas as pd
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import time


# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Drop the 'RowNumber' column as it's not needed
df = df.drop('RowNumber', axis=1)

# Display the first few rows
df.head()

# Create feature dataset (X) and target variable (y)
X = df.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

# Handle categorical variables using one-hot encoding
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)  # Convert 'Gender' to numerical (Male=1, Female=0)
X = pd.get_dummies(X, columns=['Geography'], drop_first=False)  # Keep all categories for 'Geography'
print(X.head())

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the training and test sets
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Initialize the ANN model
ann = tf.keras.models.Sequential()

# First layer - Input + first hidden layer
# shape of the input is (7000, 12)
ann.add(tf.keras.layers.Dense(units=10, activation='relu', input_shape=(12,)))

# Second layer - Hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Third layer - Output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the ANN model
ann.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN model
ann.fit(scaled_X_train, y_train, batch_size=32, epochs=100)

# Import necessary libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on the test set
y_pred = (ann.predict(scaled_X_test) > 0.5)

# Print evaluation metrics
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

ann.save("ann_churn.h5")
joblib.dump(scaler, 'scaler.pkl')

print('start')
time0 = time.perf_counter()
ann = load_model("ann_churn.h5")
scaler = joblib.load('scaler.pkl')

print(f"{time.perf_counter() - time0:.6}")

# Predicting on new customer data
new_customer_details = [[600, 40, 3, 60000, 2, 1, 1, 50000, 1, 1, 0, 0]]

# Scale the new data
scaled_customer_details = scaler.transform(new_customer_details)

# Predict whether the customer will exit (1) or stay (0)
print(ann.predict(scaled_customer_details) > 0.5)