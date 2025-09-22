## Developing a Neural Network Classification Model

## AIM:

To develop a neural network classification model for the given dataset.

## THEORY:

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model:

<img width="877" height="680" alt="image" src="https://github.com/user-attachments/assets/542cf6b6-0b80-4640-8e4b-b95fd5fd364c" />

## DESIGN STEPS:

**STEP 1:** Import necessary libraries.

**STEP 2:** Load the dataset "customers.csv"

**STEP 3:** Analyse the dataset and drop the rows which has null values.Analyse the dataset and drop the rows which has null values.

**STEP 4:** Use encoders and change the string datatypes in the dataset.

**STEP 5:** Calculate correlation matrix ans plot heatmap and analyse the data.

**STEP 6:** Use various visualizations like pairplot,displot,countplot,scatterplot and visualize the data.

**STEP 7:** Split the dataset into training and testing data using train_test_split.

**STEP 8:** Create a neural network model with 2 hidden layers and output layer with four neurons representing multi-classification.
**STEP 9:** Compile and fit the model with the training data
**STEP 10:** Validate the model using training data.
**STEP 11:** Evaluate the model using confusion matrix.

**PROGRAM**

**Name:** YOGAVARMA B

**Register Number:** 2305002029
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
import pickle
from tensorflow.keras.layers import Dense
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pylab as plt

# ------------------------------
# Load dataset
# ------------------------------
customer_df = pd.read_csv('/content/customers.csv')

# Drop missing values
customer_df_cleaned = customer_df.dropna(axis=0)

# ------------------------------
# Encoding categorical columns
# ------------------------------
categories_list = [
    ['Male', 'Female'],
    ['No', 'Yes'],
    ['No', 'Yes'],
    ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
     'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
    ['Low', 'Average', 'High']
]

# Ordinal encoding for ordered categories
enc = OrdinalEncoder(categories=categories_list, handle_unknown="use_encoded_value", unknown_value=-1)
customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
             'Graduated',
             'Profession',
             'Spending_Score']] = enc.fit_transform(
    customers_1[['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']]
)

# Label encoding for target
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

# Drop unnecessary columns
customers_1 = customers_1.drop(['ID', 'Var_1'], axis=1)

# ------------------------------
# Exploratory plots
# ------------------------------
corr = customers_1.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="BuPu", annot=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Family_Size', y='Spending_Score', data=customers_1)

# ------------------------------
# One-hot encoding target
# ------------------------------
y1 = customers_1['Segmentation'].values.reshape(-1, 1)
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y = one_hot_enc.transform(y1).toarray()

# Features
X = customers_1.drop('Segmentation', axis=1)

# ------------------------------
# Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=50
)

# Convert DataFrame → Numpy arrays
X_train = X_train.values
X_test = X_test.values

# ------------------------------
# Scale "Age" column (assuming column index 2 is Age)
# ------------------------------
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:, 2].reshape(-1, 1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:, 2] = scaler_age.transform(X_train[:, 2].reshape(-1, 1)).reshape(-1)
X_test_scaled[:, 2] = scaler_age.transform(X_test[:, 2].reshape(-1, 1)).reshape(-1)

# ------------------------------
# Model definition
# ------------------------------
model = Sequential([
    Dense(units=8, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(units=16, activation='relu'),
    Dense(units=4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# Train model
# ------------------------------
history = model.fit(
    x=X_train_scaled, y=y_train,
    epochs=200,  # reduced from 2000
    batch_size=256,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# ------------------------------
# Metrics
# ------------------------------
metrics = pd.DataFrame(history.history)
metrics[['loss', 'val_loss']].plot()

predi = model.predict(X_test_scaled)
x_test_predictions = np.argmax(predi, axis=1)
y_test_truevalue = np.argmax(y_test, axis=1)

print(confusion_matrix(y_test_truevalue, x_test_predictions))
print(classification_report(y_test_truevalue, x_test_predictions))

# ------------------------------
# Save model & preprocessing objects
# ------------------------------
model.save('customer_classification_model.h5')

with open('customer_data.pickle', 'wb') as fh:
    pickle.dump([X_train_scaled, y_train,
                 X_test_scaled, y_test,
                 customers_1, customer_df_cleaned,
                 scaler_age, enc, one_hot_enc, le], fh)

# ------------------------------
# Reload and test single prediction
# ------------------------------
model = load_model('customer_classification_model.h5')

with open('customer_data.pickle', 'rb') as fh:
    [X_train_scaled, y_train,
     X_test_scaled, y_test,
     customers_1, customer_df_cleaned,
     scaler_age, enc, one_hot_enc, le] = pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2, :]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))

~~~

**Dataset Information**
<img width="834" height="195" alt="{7F59C1FC-A1C4-4D52-9E92-468FB4C0ABCC}" src="https://github.com/user-attachments/assets/603ced3c-2ee0-47bd-a633-ceb0fcfe897b" />


**OUTPUT**
Training Loss, Validation Loss Vs Iteration Plot:
<img width="615" height="435" alt="{1128A500-D086-4A3D-829D-D42FFD42A571}" src="https://github.com/user-attachments/assets/8bc2a0b1-1e82-4717-839b-c8ac2e3e5873" />

**Confusion Matrix**
<img width="712" height="288" alt="{7656C955-56A4-4C64-A21F-A11E02CE0983}" src="https://github.com/user-attachments/assets/17b13f8c-8709-4873-be05-97dd34179e1c" />


**Classification Report**
<img width="595" height="151" alt="{17EB8532-B854-4E8C-9A57-B22D544EB767}" src="https://github.com/user-attachments/assets/6a11bc9f-f5eb-4a97-a2ac-5fb7a66b4080" />


**New Sample Data Prediction**
<img width="864" height="325" alt="{805450BC-5053-4FA6-BA50-64B1741C230E}" src="https://github.com/user-attachments/assets/56416af5-3f13-4085-ae72-c103312b0760" />


**RESULT**
Thus a neural network classification model is developed for the given dataset.

Include your result here

