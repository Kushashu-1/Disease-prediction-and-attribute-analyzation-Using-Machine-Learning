import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures


# ----------------------------Navigation Bar-------------------------------------#

st.sidebar.title(
    "Disease Prediction and Diagnostic Analysis Using Deep Learning")
page = st.sidebar.radio(" ", ('Home', 'Dataset Analysis', 'About'))

#----------------------------------About Page-----------------------------------#
if page == 'About':
    st.title("About Project and Creator")
    st.write(
        'I approached this project to learn about the Core Concept of Machine learning environment Under the Guidance : Dr. Rohit Gupta Sir . With this in mind,I have learned Machine Learing project lifecycle .Studied about different datasets and algorithms of Machine learning and also got idea how different values of parameter effects performance of Algorithm . For Better User Interface experience and designs I taken help of Streamlit.io which is open source app framework which supports Python language.'
    )
    st.subheader('Aim')
    st.write('Aim of the Project is to Analyse Differnt parameters of Disease datasets and analyzes the data and show performance for the various classification algorithms used i.e ')
    st.write('1. Logistic Regression')
    st.write('2. Random Forest ')
    st.write('3. K-Nearest Neighbor')

    st.subheader('Datasets Used :')
    st.write('1 . Breast Cancer Dataset : Worldwide, breast cancer is the most common type of cancer in women and the second highest in terms of mortality rates.Diagnosis of breast cancer is performed when an abnormal lump is found (from self-examination or x-ray) or a tiny speck of calcium is seen(on an x-ray). After a suspicious lump is found, the doctor will conduct a diagnosis to determine whether it is cancerous and,if so, whether it has spread to other parts of the body.This breast cancer dataset was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg.')
    st.write('For more Information : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29')
    st.write('2 . Cardiovascular Disease dataset : Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases,such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you''re born with (congenital heart defects),among others. The term "heart disease" is often used interchangeably with the term "cardiovascular disease." Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke. Other heart conditions, such as those that affect your heart''s muscle, valves or rhythm, also are considered forms of heart disease. We have a data which  classified if patients have heart disease or not according to features in it.')
    st.write('For More Information : https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?resource=download')
    st.write('2 . This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. All the patients here are female 21 years or older.It contains the following columns: Pregnancies: Number of times pregnant Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test BloodPressure: Diastolic blood pressure (mm Hg) SkinThickness: Triceps skin fold thickness (mm)Insulin: 2-Hour serum insulin (mu U/ml) BMI: Body mass index (weight in kg/(height in m)^2)DiabetesPedigreeFunction: Diabetes pedigree function Age: Age (years) Outcome: Class variable (0 or 1)')
    st.write(
        'For more Information :https://www.kaggle.com/datasets/mathchi/diabetes-data-set')

    st.subheader("Created By : ")
    st.write("Ashutosh Singh Kushwaha  Contact at : https://in.linkedin.com/in/ashutosh-singh-kushwaha-0836b5182")


#------------------------------------HOME---------------------------------------#

elif page == 'Home':
    st.write("""
    ## Select DataSet which you would like to Analyze ?
    """)
    selected_dataset = st.selectbox(
        '',
        ('Diabetes Dataset', 'Breast Cancer Dataset', 'Cardiovascular Disease dataset'))

    st.write("""
    ## Select any Classifier Algorithm 
    """)
    selected_classifier = st.selectbox(
        'Classifier', ('K Nearest Neighbor (KNN)', 'Logistic Regression', 'Random Forest'))
    if selected_dataset == 'Diabetes Dataset':
        df = pd.read_csv('data/diabetes.csv')
    elif selected_dataset == 'Cardiovascular Disease dataset':
        df = pd.read_csv('data/CardioT.csv')
    elif selected_dataset == 'Breast Cancer Dataset':
        df = pd.read_csv('data/Breast_cancer_data.csv')

    st.subheader('Dataset Preview')
    st.write(df.head(5))
    if selected_dataset == 'Breast Cancer Dataset':
        df.drop('Unnamed: 32', axis=1, inplace=True)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df['target'] = df.diagnosis
        df.drop(columns=['diagnosis'], inplace=True)

    elif selected_dataset == 'Cardiovascular Disease dataset':
        df['target'] = df.cardio
        df.drop(columns=['cardio'], inplace=True)
    else:
        df['target'] = df.target

    y = df['target']
    x = df.drop(columns=['target'])
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(['target'], axis='columns'), y, test_size=0.2)
    st.write("Testing Data Size: ", len(x_test))
    st.write("Training Data Size ", len(x_train))

    if selected_classifier == 'Random Forest':
        max_depth = st.sidebar.slider("Maximum Depth", 2, 15, 6)
        n_estimators = st.sidebar.slider("Number of Estimeter", 1, 100, 15)
        model = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators)

    elif selected_classifier == 'K Nearest Neighbor (KNN)':
        n_neighbors = st.sidebar.slider("Number of neighbors", 1, 100, 6)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    elif selected_classifier == 'Logistic Regression':
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, 200)
        model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)

    model.fit(x_train, y_train)
    # st.write("Accuracy of Model", model.score(x_test, y_test))
    y_predicted = model.predict(x_test)
    st.write("Accuracy Score of Model (%)",
             accuracy_score(y_predicted, y_test)*100)
    plot = confusion_matrix(y_test, y_predicted)
    st.write(plot)
    f = plt.figure(figsize=(12, 12))
    fig = sns.heatmap(plot, annot=True)
    fig.set_xlabel('Truth Value')
    fig.set_ylabel('Predicted Value')
    st.pyplot(f)

#---------------------------------DataSet Analysis-------------------------------#

else:
    st.write("""
    # Which dataset would you like to know about?
    """)
    selected_dataset = st.selectbox(
        '', ('Diabetes Dataset', 'Breast Cancer Dataset', 'Cardiovascular Disease dataset'))

    y = ''
    if selected_dataset == 'Diabetes Dataset':
        loaded_data = pd.read_csv('data/diabetes.csv')
        y = 'target'
    elif selected_dataset == 'Breast Cancer Dataset':
        loaded_data = pd.read_csv('data/Breast_cancer_data.csv')
        y = 'diagnosis'
    else:
        loaded_data = pd.read_csv('data/CardioT.csv')
        y = 'cardio'

    if st.button("View DataSet"):
        st.dataframe(loaded_data)
    if st.button("Show All Features"):
        features = pd.DataFrame(loaded_data.columns)
        features.columns = ['Features']
        st.dataframe(data=features, height=500)
        st.write("Shape of DataSet", loaded_data.shape)
        st.write("number of Classes",
                 loaded_data[y].value_counts())

    options_list = []
    if selected_dataset == 'Diabetes Dataset':
        options_list = ['Scatter Matrix',
                        'Age Vs Glucose', 'Insulin Vs BMI', 'Heatmap']
    elif selected_dataset == 'Breast Cancer Dataset':
        options_list = ['Scatter Matrix', 'Number of Malignant and Benign',
                        'Heatmap', 'Mean smoothness vs Mean area']
    elif selected_dataset == 'Cardiovascular Disease dataset':
        options_list = ['Scatter Matrix', 'Age Vs Systolic blood pressure',
                        'Diastolic blood pressure Vs Alcohol intake', 'Heatmap']

    plots = st.multiselect("Graphical Representation", options_list)
    if st.button("Plot", key='Graphs'):
        if selected_dataset == 'Breast Cancer Dataset':
            df = pd.read_csv('data/Breast_cancer_data.csv')
            df.iteritems = df.items
            if 'Number of Malignant and Benign' in plots:
                st.subheader("Malignant and Benign Count")
                fig, ax = plt.subplots()
                b = len(df[df['diagnosis'] == 'B'])
                m = len(df[df['diagnosis'] == 'M'])
                count = [b, m]
                bars = plt.bar(np.arange(2), count, color=[
                               '#000099', '#ffff00'])
                # show value in bars
                for bar in bars:
                    height = bar.get_height()
                    plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90,
                                   '{0:.{1}f}'.format(height, 2), ha='center', color='black', fontsize=11)
                plt.tick_params(top='off', bottom='off', left='off',
                                right='off', labelleft='on', labelbottom='on')
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                plt.xticks(ticks=[0, 1])
                ax.set_ylabel('Count')
                ax.set_xlabel('Target')
                ax.xaxis.set_tick_params(length=0)
                ax.yaxis.set_tick_params(length=0)
                st.pyplot(fig)

            if 'Scatter Matrix' in plots:
                st.subheader("Scatter Matrix")
                fig = px.scatter_matrix(df, dimensions=[
                                        'radius_mean', 'texture_mean', 'perimeter_mean', 'radius_worst', 'texture_worst', 'perimeter_worst'], color="diagnosis", width=800, height=700)
                st.write(fig)

            if 'Heatmap' in plots:
                st.subheader("Heatmap")
                fig = plt.figure(figsize=(30, 20))
                hmap = sns.heatmap(df.drop(columns=['diagnosis']).corr(
                ), annot=True, cmap='Blues', annot_kws={"size": 18})
                hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=25)
                hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=25)
                st.pyplot(fig)

            if 'Mean smoothness vs Mean area' in plots:
                st.subheader('Cancer Smoothness and Area')
                fig = plt.figure()
                sns.scatterplot(x=df['smoothness_mean'], y=df['area_mean'],
                                hue=df['diagnosis'], palette=['#000099', '#ffff00'])
                st.pyplot(fig)

        elif selected_dataset == 'Diabetes Dataset':
            df = pd.read_csv('data/diabetes.csv')
            df.iteritems = df.items
            if 'Scatter Matrix' in plots:
                st.subheader("Scatter Matrix")
                fig = px.scatter_matrix(df, dimensions=[
                                        'Pregnancies', 'Glucose', 'BMI', 'target'], color="target", width=800, height=700)
                st.write(fig)
            if 'Age Vs Glucose' in plots:
                st.subheader("Age Vs Glucose")
                fig = plt.figure()
                sns.scatterplot(x=df['Age'], y=df['Glucose'], hue=df['target'], palette=[
                                '#fc2803', '#fce803'])
                st.pyplot(fig)
            if 'Insulin Vs BMI' in plots:
                st.subheader("Insulin Vs BMI")
                fig = plt.figure()
                sns.scatterplot(x=df['Insulin'], y=df['BMI'], hue=df['target'], palette=[
                                '#03fc30', '#fc03a1'])
                st.pyplot(fig)
            if 'Heatmap' in plots:
                st.subheader("Heatmap")
                fig = plt.figure(figsize=(10, 10))
                hmap = sns.heatmap(df.drop(columns=['target']).corr(
                ), annot=True, cmap='Blues', annot_kws={"size": 18})
                hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=25)
                hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=25)
                st.pyplot(fig)

#---------------------------------------------Heart-----------------------------------------------------#
        elif selected_dataset == 'Cardiovascular Disease dataset':
            st.write("Large Dataset ")
            df = pd.read_csv('data/CardioT.csv')
            df['target'] = df.cardio
            df.iteritems = df.items
            if 'Scatter Matrix' in plots:
                st.subheader("Scatter Matrix")
                fig = px.scatter_matrix(df, dimensions=[
                                        'age', 'height', 'ap_hi', 'cholesterol', 'gluc', 'smoke', 'target'], color="target", width=800, height=700)
                st.write(fig)

            if 'Age Vs Systolic blood pressure' in plots:
                st.subheader("Age Vs Systolic blood pressure")
                fig = plt.figure()
                p = sns.scatterplot(x=df['age'], y=df['ap_hi'], hue=df['target'], palette=[
                                    '#fc2803', '#fce803'])
                p.set_xlabel("Age")
                p.set_ylabel("Systolic blood pressure")
                st.pyplot(fig)
            if 'Diastolic blood pressure Vs Alcohol intake' in plots:
                st.subheader("Diastolic blood pressure Vs Alcohol intake")
                fig = plt.figure()
                p = sns.scatterplot(x=df['ap_lo'], y=df['alco'], hue=df['target'], palette=[
                                    '#03fc30', '#fc03a1'])
                p.set_xlabel("Diastolic blood pressure")
                p.set_ylabel("Alcohol intake")
                st.pyplot(fig)
            if 'Heatmap' in plots:
                st.subheader("Heatmap")
                fig = plt.figure(figsize=(40, 40))
                hmap = sns.heatmap(df.drop(columns=['target']).corr(
                ), annot=True, cmap='Blues', annot_kws={"size": 18})
                hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=25)
                hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=25)
                st.pyplot(fig)
