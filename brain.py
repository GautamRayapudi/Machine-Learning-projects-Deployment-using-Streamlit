import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.outliers import Winsorizer
import pickle

class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, capping_method='gaussian', tail='both', fold=3, variables=None):
        self.cols = cols
        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold
        self.variables = variables
        self.winsorizers = {}

    def fit(self, X, y=None):
        self.winsorizers = {}
        if self.cols is None:
            self.cols = X.columns

        for col in self.cols:
            winsorizer = Winsorizer(capping_method=self.capping_method, tail=self.tail, fold=self.fold)
            winsorizer.fit(X[[col]])
            self.winsorizers[col] = winsorizer
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)
        for col, winsorizer in self.winsorizers.items():
            X_transformed[col] = winsorizer.transform(X[[col]]).squeeze()
        return X_transformed

st.title("Brain Tumor Prediction using ML")
inputs = []
result = None

Area = st.number_input('Area')
inputs.append(Area)
Perimeter = st.number_input('Perimeter')
inputs.append(Perimeter)
Convex_Area = st.number_input('Convex Area')
inputs.append(Convex_Area)
Solidity = st.number_input('Solidity')
inputs.append(Solidity)
Equivalent_Diameter = st.number_input('Equivalent Diameter')
inputs.append(Equivalent_Diameter)
Major_Axis = st.number_input('Major Axis')
inputs.append(Major_Axis)
Minor_Axis = st.number_input('Minor Axis')
inputs.append(Minor_Axis)
Eccentricity = st.number_input('Eccentricity')
inputs.append(Eccentricity)

model = pickle.load(open(r"C:\Users\ASUS\Documents\Innomatics Jupyter notebooks\Machine Learning\Snake\Tasks\Tuesday Task\tumor.pkl","rb"))

if st.button("Submit") == True:
    inputs_df = pd.DataFrame([inputs], columns=['Area', 'Perimeter', 'Convex Area', 'Solidity', 'Equivalent Diameter', 'Major Axis', 'Minor Axis', 'Eccentricity'])
    result = model.predict(inputs_df)[0]
    if result == 0:
        st.text("No Tumor Detected")
    elif result == 1:
        st.text("Tumor Detected")
