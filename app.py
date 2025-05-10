import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew, kurtosis

# Set page configuration
st.set_page_config(page_title="UFO Sightings Analysis", layout="wide")

# Title and description
st.title("UFO Sightings Analysis Dashboard")
st.markdown("Explore UFO sightings data with dynamic visualizations and machine learning models.")

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Data Overview", "Visualizations", "Machine Learning", "Map"])

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('scrubbed.csv')
    df.columns = df.columns.str.strip()
    
    # Data cleaning
    df['datetime'] = df['datetime'].astype(str).str.replace('24:00', '00:00')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')
    df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'], errors='coerce')
    df['date posted'] = pd.to_datetime(df['date posted'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Fill missing values
    df['state'].fillna('Unknown', inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['shape'].fillna('Unknown', inplace=True)
    df['duration (seconds)'].fillna(0, inplace=True)
    df['comments'].fillna('', inplace=True)
    df['latitude'].fillna(df['latitude'].mean(), inplace=True)
    df['longitude'].fillna(df['longitude'].mean(), inplace=True)
    
    # Remove outliers
    for col in ['duration (seconds)', 'latitude', 'longitude']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

df = load_data()

# Data Overview Mode
if app_mode == "Data Overview":
    st.header("Data Overview")
    st.write("Dataset Info:")
    st.write(df.info())
    
    st.subheader("Null Values")
    null_values = pd.concat([df.isnull().sum(), round((df.isnull().sum() / len(df)) * 100, 2)], 
                            axis=1, keys=['Number of Null Values', 'Percentage of Null Values'])
    st.dataframe(null_values)
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

# Visualizations Mode
elif app_mode == "Visualizations":
    st.header("Visualizations")
    
    # Dynamic column selection
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_col = st.selectbox("Select Numeric Column for Visualization", num_cols)
    
    # Histogram
    st.subheader(f"Histogram of {selected_col}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], bins=30, kde=True, ax=ax)
    skewness = skew(df[selected_col].dropna())
    kurt = kurtosis(df[selected_col].dropna())
    ax.set_title(f'Distribution of {selected_col}\nSkewness: {skewness:.2f}, Kurtosis: {kurt:.2f}')
    st.pyplot(fig)
    
    # Box Plot
    st.subheader(f"Box Plot of {selected_col}")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=selected_col, ax=ax)
    ax.set_title(f'Box Plot of {selected_col}')
    st.pyplot(fig)
    
    # UFO Shapes Distribution
    st.subheader("UFO Shapes Distribution")
    shape_counts = df['shape'].value_counts().reset_index()
    shape_counts.columns = ['Shape', 'Count']
    fig = px.bar(shape_counts, x='Shape', y='Count', color='Shape', title='Distribution of UFO Shapes')
    st.plotly_chart(fig)
    
    # Sightings by Year
    st.subheader("UFO Sightings by Year")
    df['year'] = df['datetime'].dt.year
    year_counts = df['year'].value_counts().sort_index().reset_index()
    year_counts.columns = ['Year', 'Count']
    fig = px.line(year_counts, x='Year', y='Count', title='Number of UFO Sightings by Year')
    st.plotly_chart(fig)

# Machine Learning Mode
elif app_mode == "Machine Learning":
    st.header("Machine Learning")
    
    # Dynamic feature selection
    features = st.multiselect("Select Features for Model", 
                             df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
                             default=['latitude', 'longitude'])
    target = 'duration (seconds)'
    
    if features:
        X = df[features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Model selection
        model_choice = st.selectbox("Select Model", 
                                    ["Linear Regression", "Decision Tree", "XGBoost", "Logistic Regression"])
        
        # Model training and evaluation
        if st.button("Train Model"):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor(max_depth=5, random_state=42)
            elif model_choice == "XGBoost":
                model = XGBRegressor(max_depth=3, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000)
                y_train = (y_train > y_train.median()).astype(int)
                y_test = (y_test > y_test.median()).astype(int)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred) if model_choice != "Logistic Regression" else model.score(X_test, y_test)
            
            st.subheader("Model Performance")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}" if model_choice != "Logistic Regression" else f"Accuracy: {r2:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='r2' if model_choice != "Logistic Regression" else 'accuracy')
            st.write(f"Cross-Validation Score (Mean): {np.mean(cv_scores):.4f}")

# Map Mode
elif app_mode == "Map":
    st.header("UFO Sightings Map")
    
    # Create Folium map
    map_ufo = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=4)
    marker_cluster = MarkerCluster().add_to(map_ufo)
    
    for index, row in df.iterrows():
        folium.Marker(location=[row['latitude'], row['longitude']],
                      popup=row['city'],
                      icon=folium.Icon(icon='cloud')).add_to(marker_cluster)
    
    # Display map
    st_folium(map_ufo, width=700, height=500)

# Dynamic calculation input
st.sidebar.header("Dynamic Calculation")
calc_col1 = st.sidebar.selectbox("Select First Column", df.columns, key="calc_col1")
calc_col2 = st.sidebar.selectbox("Select Second Column", df.columns, key="calc_col2")
operation = st.sidebar.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])

if st.sidebar.button("Calculate"):
    if calc_col1 and calc_col2:
        try:
            if operation == "Add":
                df['calculated_column'] = df[calc_col1] + df[calc_col2]
            elif operation == "Subtract":
                df['calculated_column'] = df[calc_col1] - df[calc_col2]
            elif operation == "Multiply":
                df['calculated_column'] = df[calc_col1] * df[calc_col2]
            elif operation == "Divide":
                df['calculated_column'] = df[calc_col1] / df[calc_col2]
            
            st.subheader("Calculated Column Preview")
            st.dataframe(df[['calculated_column']].head())
        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")