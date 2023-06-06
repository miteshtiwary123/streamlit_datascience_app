from PIL import Image
import streamlit as st
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

        if data is not None:
            st.success('Data successfully loaded')
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

        if data is not None:
            st.success('Data successfully loaded')
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

    # Building Model
    elif option == 'model':
        st.subheader("Model Building")
        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success("Data successfully loaded")
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            if st.checkbox('Select Multiple columns'):
                new_data = st.multiselect(
                    "Select your preferred columns. NB: Let your target variable be the last column to be selected",
                    df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                # Dividing my data into X and y variables
                X = df1.iloc[:, 0:-1]
                y = df1.iloc[:, -1]

            seed = st.sidebar.slider('Seed', 1, 200)
            classifier_name = st.sidebar.selectbox('Select your preferred classifier:',
                                                   ('KNN', 'SVM', 'LR', 'naive_bayes', 'decision tree'))

            def add_parameter(name_of_clf):
                params = dict()
                if name_of_clf == 'SVM':
                    C = st.sidebar.slider('C', 0.01, 15.0)
                    params['C'] = C
                else:
                    name_of_clf == 'KNN'
                    K = st.sidebar.slider('K', 1, 15)
                    params['K'] = K
                    return params

            # calling the function
            params = add_parameter(classifier_name)

            # defining a function for our classifier
            def get_classifier(name_of_clf, params):
                clf = None
                if name_of_clf == 'SVM':
                    clf = SVC(C=params['C'])
                elif name_of_clf == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf == 'LR':
                    clf = LogisticRegression()
                elif name_of_clf == 'naive_bayes':
                    clf = GaussianNB()
                elif name_of_clf == 'decision tree':
                    clf = DecisionTreeClassifier()
                else:
                    st.warning('Select your choice of algorithm')

                return clf

            new_clf = get_classifier(classifier_name, params)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            new_clf.fit(X_train, y_train)

            y_pred = new_clf.predict(X_test)
            st.write('Predictions:', y_pred)

            accuracy = accuracy_score(y_test, y_pred)

            st.write('Name of classifier:', classifier_name)
            st.write('Accuracy', accuracy)

            # DEALING WITH THE ABOUT US PAGE

    elif option == 'About_us':

        st.markdown(
            'This is an interactive web page for our ML project, feel free to use it. '
            'The analysis in here is to demonstrate how we can present our wok '
            'to our stakeholders in an interactive way by building a web app '
            'for our machine learning algorithms using different dataset.'
            )

        st.balloons()


if __name__ == '__main__':
    main()

