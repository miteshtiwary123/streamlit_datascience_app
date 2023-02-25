from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
matplotlib.use('Agg')

# Set Title
st.title('My Data Science Project')
image = Image.open('mitesh1.png')
st.image(image, use_column_width=True)


def main():
    activities = ['EDA', 'Visualisation', 'model', 'About_us']
    option = st.sidebar.selectbox('Selection option:', activities)

    # Dealing with EDA part
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        data = st.file_uploader('Upload dataset:', type=['csv', 'xlsx', 'txt', 'json'])
        st.success('Data successfully loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_columns = st.multiselect("Select preferred columns:", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
            if st.checkbox("Display summary"):
                st.write(df.describe().T)
            if st.checkbox("Display Null Values"):
                st.write(df.isnull().sum())
            if st.checkbox("Display the data types"):
                st.write(df.dtypes)
            if st.checkbox("Display Correlation"):
                st.write(df.corr())

    # Dealing with visualization
    elif option == 'Visualisation':
        st.subheader("Data Visualization")
        data = st.file_uploader('Upload dataset:', type=['csv', 'xlsx', 'txt', 'json'])
        st.success('Data successfully loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            if st.checkbox('Select Multiple columns to plot'):
                selected_columns = st.multiselect("Select preferred columns:", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Heatmap'):
                fig = plt.figure()
                st.write(sns.heatmap(df1.corr(), vmax=1, square=True, annot=True, cmap='viridis'))
                st.pyplot(fig)
            if st.checkbox('Display Pair plot'):
                fig = sns.pairplot(df, diag_kind='kde')
                st.write(fig)
                st.pyplot(fig)
            if st.checkbox('Display pie chart'):
                fig = plt.figure()
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox("Select columns to display", all_columns)
                pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pieChart)
                st.pyplot(fig)




if __name__ == '__main__':
    main()

