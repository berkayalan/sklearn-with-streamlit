import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,ShuffleSplit,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,BaseEnsemble,GradientBoostingClassifier
from sklearn.svm import SVC,LinearSVC
import time
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets


@st.cache
def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data = datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name=="Wine Dataset":
        data = datasets.load_wine()
    X = data.data
    y=data.target
    return X,y


def parameter_changes(clf_name):
    params = dict()
    if clf_name=="KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"]=K
    elif clf_name=="Logistic Regression":
        penalty = st.sidebar.selectbox("penalty",("l2","none"))
        params["penalty"]=penalty
    elif clf_name=="XGBoost":
        learning_rate = st.sidebar.slider("learning_rate",0.01,10.0)
        params["learning_rate"]=learning_rate
    elif clf_name=="Random Forests":
        n_estimators = st.sidebar.slider("n_estimators",0,1500,100)
        params["n_estimators"]=n_estimators
    elif clf_name=="Decision Tree":
        criterion=st.sidebar.selectbox("criterion",("gini","entropy"))
        max_depth = st.sidebar.slider("max_depth",1,15)
        params["max_depth"]=max_depth
        params["criterion"]=criterion
    elif clf_name=="Support Vector Machines":
        C = st.sidebar.slider("C",1,15)
        kernel=st.sidebar.selectbox("kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"]=C
        params["kernel"]=kernel
    
    return params

def get_algorithm(algorithm,params):
    if algorithm=="KNN":
        alg = KNeighborsClassifier(n_neighbors=params["K"])
    elif algorithm=="Logistic Regression":
        alg = LogisticRegression(penalty=params["penalty"])
    elif algorithm=="XGBoost":
        alg = XGBClassifier(learning_rate=params["learning_rate"])
    elif algorithm=="Random Forests":
        alg = RandomForestClassifier(n_estimators=params["n_estimators"])
    elif algorithm=="Decision Tree":
        alg = DecisionTreeClassifier(max_depth=params["max_depth"],criterion=params["criterion"])
    elif algorithm=="Support Vector Machines":
        alg = SVC(C=params["C"],kernel=params["kernel"])
    return alg
       


##########

titles = st.container()
dataset = st.container()
training = st.container()
prediction = st.container()
visualisation = st.container()

with titles:
    st.title("Welcome to Sklearn Datasets Machine Learning Classification Project")
    image = Image.open('sklearn.png')
    st.image(image, caption='Sklearn Logo')

    st.write("""
    # Let's explore Scikit-Learn First

    scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.

    Check the official main page here: https://scikit-learn.org/stable/

    """)

with dataset:

    st.title("Dataset Information")

    dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))
    #Check out all available dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html

    st.sidebar.subheader("{0} dataset selected!".format(dataset_name))
    st.write("{0} dataset selected!".format(dataset_name))


    algorithm_name = st.sidebar.selectbox("Select Algorithm",("Logistic Regression","Random Forests","XGBoost","Decision Tree","KNN","Support Vector Machines"))

    st.sidebar.subheader("{0} algorithm selected!".format(algorithm_name))
    st.write("{0} algorithm selected!".format(algorithm_name))

    X,y = get_dataset(dataset_name)
    

    st.sidebar.subheader("Arrange the parameters!")

    params = parameter_changes(algorithm_name)

    st.write("Shape of Predictor dataset: ",X.shape)
    st.write("Number of Classes: ",len(np.unique(y)))
    st.write("Let's look at the predictor data!")
    st.write(X[:5])

    st.write("Let's look at the data that we will predict!")
    st.write(y[:5])

with training:
    st.title("Let's train the model!")
    alg_name= get_algorithm(algorithm_name,params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    start_time = time.time()
    alg_name.fit(X_train,y_train)
    elapsed_time = time.time() - start_time
    st.write("Elapsed time for Model Training: {} seconds".format(elapsed_time))
    
with prediction:
    st.title("Let's predict with the model!")
    y_pred= alg_name.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {algorithm_name}')
    st.write(f'Accuracy =', acc)

with visualisation:
    st.title("Let's project the data onto the 2 primary principal component")
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    #plt.show()
    st.pyplot(fig)
