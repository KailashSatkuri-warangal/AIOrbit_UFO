# AIOrbit_UFO

![AIOrbit_UFO](https://img.shields.io/badge/Streamlit-App-blue)  
![GitHub Stars](https://img.shields.io/github/stars/KailashSatkuri-warangal/AIOrbit_UFO?style=social)  

An AI-powered web application for dynamic CSV data analysis, pattern discovery, and predictive modeling. Built with Streamlit, it supports custom CSV uploads, interactive visualizations, clustering, anomaly detection, and machine learning, with a default UFO sightings dataset.

## Overview
AIOrbit_UFO is a versatile platform for exploring and analyzing structured datasets. Whether you're investigating UFO sightings or your own CSV data, this app provides tools for:
- **Dynamic Data Upload**: Analyze any CSV with automatic preprocessing.
- **Pattern Finding**: Discover clusters and anomalies using AI techniques.
- **Machine Learning**: Train predictive models with user-selected features.
- **Interactive Visualizations**: Explore data through plots and geographic maps.

Developed by Kailash Satkuri, a Machine Learning enthusiast from Warangal, India, this project showcases the power of AI in data analysis.

## Features
- **Dynamic CSV Upload**: Upload custom CSVs or use the default UFO sightings dataset (`scrubbed.csv`).
- **Pattern Discovery**:
  - KMeans clustering with user-defined features and cluster counts.
  - Anomaly detection using z-scores with adjustable thresholds.
- **Machine Learning**:
  - Train models: Linear Regression, Decision Tree, XGBoost, Logistic Regression.
  - Dynamic feature and target selection.
  - Real-time performance metrics (R², MSE, cross-validation).
- **Visualizations**:
  - Histograms, box plots, and correlation matrices.
  - Interactive Plotly charts for clusters and trends.
  - Folium-based geographic maps for spatial data.
- **Dynamic Calculations**: Perform real-time column operations (add, subtract, multiply, divide).
- **Exportable Results**: Download processed datasets as CSV.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KailashSatkuri-warangal/AIOrbit_UFO.git
   cd AIOrbit_UFO