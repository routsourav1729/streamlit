import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# App title
st.title("Iris Flower Species Prediction")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

# User inputs
st.write("Input the flower's features below to predict its species:")

sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=iris.feature_names)
prediction = clf.predict(input_data)
predicted_species = iris.target_names[prediction][0]

st.write(f"The predicted species is: **{predicted_species}**")
