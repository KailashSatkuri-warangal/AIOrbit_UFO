import streamlit as IllegalStateException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from scipy.stats import skew, kurtosis
import io
import base64
import streamlit as st
# Set page configuration
st.set_page_config(page_title="AI-Powered Data Analysis", layout="wide")

# Title and description
st.title("AI-Powered Data Analysis Platform")
st.markdown("Upload a CSV file or use the default UFO dataset to explore, analyze, and discover patterns with AI-driven tools.")

# Sidebar for navigation and data upload
st.sidebar.header("Data Input")
data_option = st.sidebar.radio("Data Source", ["Use Default UFO Dataset", "Upload CSV File"])

# Load data
@st.cache_data
def load_default_data():
    df = pd.read_csv('scrubbed.csv')
    df.columns = df.columns.str.strip()
    df['datetime'] = df['datetime'].astype(str).str.replace('24:00', '00:00')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')
    df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'], errors='coerce')
    df['date posted'] = pd.to_datetime(df['date posted'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['state'].fillna('Unknown', inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['shape'].fillna('Unknown', inplace=True)
    df['duration (seconds)'].fillna(0, inplace=True)
    df['comments'].fillna('', inplace=True)
    df['latitude'].fillna(df['latitude'].mean(), inplace=True)
    df['longitude'].fillna(df['longitude'].mean(), inplace=True)
    return df

# Handle CSV upload
def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        # Attempt to convert potential datetime columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            # Convert numeric columns
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

# Load data based on user choice
if data_option == "Use Default UFO Dataset":
    df = load_default_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Navigation modes
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Data Overview", "Visualizations", "Pattern Finding", "Machine Learning", "Map"])

# Data Overview Mode
if app_mode == "Data Overview":
    st.header("Data Overview")
    st.write("Dataset Info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.subheader("Null Values")
    null_values = pd.concat([df.isnull().sum(), round((df.isnull().sum() / len(df)) * 100, 2)], 
                            axis=1, keys=['Number of Null Values', 'Percentage of Null Values'])
    st.dataframe(null_values)
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
    
    # Download processed data
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
    st.markdown(href, unsafe_allow_html=True)

# Visualizations Mode
elif app_mode == "Visualizations":
    st.header("Visualizations")
    
    # Dynamic column selection
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_col = st.selectbox("Select Numeric Column for Visualization", num_cols)
    
    # Histogram
    st.subheader(f"Histogram of {selected_col}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col].dropna(), bins=30, kde=True, ax=ax)
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
    
    # Correlation Matrix
    if len(num_cols) > 1:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

# Pattern Finding Mode
elif app_mode == "Pattern Finding":
    st.header("Pattern Finding with AI")
    
    # Select features for clustering
    cluster_features = st.multiselect("Select Features for Clustering", 
                                     df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
                                     default=df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:2])
    
    if cluster_features and len(cluster_features) >= 2:
        X_cluster = df[cluster_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Dynamic number of clusters
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        if st.button("Run Clustering"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = pd.Series(index=df.index, dtype=float)
            df.loc[X_cluster.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
            
            st.subheader("Clustering Results")
            st.write(f"Silhouette Score: {silhouette_score(X_scaled, df.loc[X_cluster.index, 'Cluster']):.4f}")
            
            # Visualize clusters
            if len(cluster_features) == 2:
                fig = px.scatter(df, x=cluster_features[0], y=cluster_features[1], color='Cluster',
                                 title="Cluster Visualization")
                st.plotly_chart(fig)
            
            # Display clustered data
            st.dataframe(df[cluster_features + ['Cluster']].head())
    
    # Anomaly Detection
    st.subheader("Anomaly Detection")
    anomaly_col = st.selectbox("Select Column for Anomaly Detection", num_cols)
    if st.button("Detect Anomalies"):
        z_scores = (df[anomaly_col] - df[anomaly_col].mean()) / df[anomaly_col].std()
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0)
        anomalies = df[abs(z_scores) > threshold]
        st.write(f"Number of Anomalies Detected: {len(anomalies)}")
        st.dataframe(anomalies[[anomaly_col]].head())

# Machine Learning Mode
elif app_mode == "Machine Learning":
    st.header("Machine Learning")
    
    # Dynamic feature and target selection
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    target = st.selectbox("Select Target Column", num_cols)
    features = st.multiselect("Select Feature Columns", 
                              [col for col in num_cols if col != target],
                              default=[col for col in num_cols if col != target][:2])
    
    if features and target:
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
            
            # Plot predictions
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

# Map Mode
elif app_mode == "Map":
    st.header("Geographic Visualization")
    
    # Check for latitude and longitude columns
    lat_col = st.selectbox("Select Latitude Column", 
                           [col for col in df.columns if 'lat' in col.lower() or df[col].dtype == 'float64'],
                           index=0 if any('lat' in col.lower() for col in df.columns) else len(df.columns)-1)
    lon_col = st.selectbox("Select Longitude Column", 
                           [col for col in df.columns if 'lon' in col.lower() or df[col].dtype == 'float64'],
                           index=1 if any('lon' in col.lower() for col in df.columns) else len(df.columns)-1)
    
    if lat_col and lon_col:
        # Create Folium map
        map_data = df[[lat_col, lon_col]].dropna()
        map_center = [map_data[lat_col].mean(), map_data[lon_col].mean()]
        map_ufo = folium.Map(location=map_center, zoom_start=4)
        marker_cluster = MarkerCluster().add_to(map_ufo)
        
        for index, row in map_data.iterrows():
            folium.Marker(location=[row[lat_col], row[lon_col]],
                          icon=folium.Icon(icon='cloud')).add_to(marker_cluster)
        
        # Display map
        st_folium(map_ufo, width=700, height=500)

# Dynamic Calculation
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