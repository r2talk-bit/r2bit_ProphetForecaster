import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
import tempfile
from PIL import Image
from datetime import datetime

# Add the project directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Prophet forecaster function
from src.wms_prophet import run_prophet_forecast

st.set_page_config(page_title="Prophet Forecaster", layout="wide")

st.title("Prophet Forecaster")

# Create a sidebar for input parameters
st.sidebar.header("Upload Data & Set Parameters")

# File upload widget in sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

# Parameters for forecasting in sidebar
st.sidebar.subheader("Forecast Parameters")

# Basic parameters
periods = st.sidebar.slider("Number of periods to forecast", min_value=7, max_value=365, value=30, step=1)
delimiter = st.sidebar.selectbox("CSV Delimiter", ["Auto-detect", ",", ";"], index=0)
delimiter = None if delimiter == "Auto-detect" else delimiter

# Time granularity
granularity = st.sidebar.selectbox(
    "Time Granularity", 
    options=["daily", "weekly", "monthly"],
    index=0,
    help="Aggregate data to daily, weekly, or monthly periods"
)

# Advanced parameters (collapsible)
with st.sidebar.expander("Advanced Parameters"):
    use_iqr = st.checkbox("Use IQR for outlier detection", value=True)
    iqr_multiplier = st.slider("IQR Multiplier", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    max_threshold = st.number_input("Max Threshold (optional)", value=None, min_value=0.0, step=1.0)
    log_transform = st.checkbox("Apply Log Transform", value=False)
    cross_validate = st.checkbox("Perform Cross-Validation", value=False)

# Main content area
if uploaded_file is not None:
    try:
        # Create a temporary directory to store outputs
        output_dir = tempfile.mkdtemp()
        
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(output_dir, "uploaded_data.csv")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display the raw data
        st.subheader("Raw Data Preview")
        
        # Try to read with auto-detection of delimiter
        try:
            if delimiter is None:
                # First, read the first few lines to check the content
                with open(temp_file_path, 'r', encoding='utf-8', errors='replace') as f:
                    sample = f.read(1024)
                
                # Count occurrences of common delimiters
                comma_count = sample.count(',')
                semicolon_count = sample.count(';')
                
                # Try to infer the delimiter based on counts
                inferred_delimiter = ',' if comma_count > semicolon_count else ';'
                st.info(f"Auto-detecting delimiter... (found {comma_count} commas and {semicolon_count} semicolons)")
                
                # Try the inferred delimiter first
                try:
                    df = pd.read_csv(temp_file_path, delimiter=inferred_delimiter)
                    if len(df.columns) > 1:
                        st.success(f"Using detected delimiter: '{inferred_delimiter}'")
                    else:
                        # If only one column, the delimiter detection might be wrong
                        # Try the other common delimiter
                        other_delimiter = ';' if inferred_delimiter == ',' else ','
                        df = pd.read_csv(temp_file_path, delimiter=other_delimiter)
                        if len(df.columns) > 1:
                            st.success(f"Using detected delimiter: '{other_delimiter}'")
                        else:
                            st.warning("Could not confidently detect delimiter. Please check your data format.")
                except Exception as e:
                    # If the inferred delimiter fails, try the other one
                    try:
                        other_delimiter = ';' if inferred_delimiter == ',' else ','
                        df = pd.read_csv(temp_file_path, delimiter=other_delimiter)
                        if len(df.columns) > 1:
                            st.success(f"Using detected delimiter: '{other_delimiter}'")
                        else:
                            st.error(f"Error with auto-detected delimiters: {str(e)}")
                            df = None
                    except Exception as e2:
                        st.error(f"Could not read file with either delimiter: {str(e2)}")
                        df = None
            else:
                # Use the user-specified delimiter
                df = pd.read_csv(temp_file_path, delimiter=delimiter)
                st.info(f"Using user-selected delimiter: '{delimiter}'")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            df = None
        
        if df is not None:
            st.dataframe(df.head())
            
            # Check for required columns
            if 'ds' not in df.columns and 'y' not in df.columns:
                st.warning("The CSV file should contain 'ds' (date) and 'y' (value) columns. If your columns have different names, please rename them.")
                
                # Attempt to help the user by showing column mapping options
                st.subheader("Column Mapping")
                st.write("Select which columns from your data correspond to the required fields:")
                
                date_col = st.selectbox("Select date column:", df.columns)
                value_col = st.selectbox("Select value column:", df.columns)
                
                if st.button("Apply Mapping"):
                        # Create a new DataFrame with the correct column names
                    df_mapped = df.copy()
                    df_mapped.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True)
                    df = df_mapped[['ds', 'y']]
                    
                    # Try to convert the date column to datetime
                    try:
                        df['ds'] = pd.to_datetime(df['ds'])
                        st.success("Column mapping applied and date format recognized!")
                    except Exception as e:
                        st.warning(f"Column mapping applied but date conversion failed: {str(e)}. Please check your date format.")
                    
                    st.dataframe(df.head())
            
            # Generate forecast button
            if st.button("Generate Forecast"):
                if 'ds' in df.columns and 'y' in df.columns:
                    with st.spinner("Generating forecast... This may take a moment."):
                        try:
                            # Call the forecasting function
                            results = run_prophet_forecast(
                                input_file=df,
                                output_dir=output_dir,
                                periods=periods,
                                delimiter=delimiter,
                                granularity=granularity,
                                use_iqr=use_iqr,
                                iqr_multiplier=iqr_multiplier,
                                max_threshold=max_threshold,
                                log_transform=log_transform,
                                cross_validate=cross_validate
                            )
                            
                            # Display results
                            st.subheader("Forecast Results")
                            
                            # Create two columns for the plots
                            col1, col2 = st.columns(2)
                            
                            # Display forecast comparison plot
                            with col1:
                                st.subheader("Forecast Comparison")
                                forecast_img = Image.open(results['plot_path'])
                                st.image(forecast_img, use_column_width=True)
                            
                            # Display components plot
                            with col2:
                                st.subheader("Forecast Components")
                                components_img = Image.open(results['components_path'])
                                st.image(components_img, use_column_width=True)
                            
                            # Display metrics
                            st.subheader("Forecast Metrics")
                            metrics = results['metrics']
                            metrics_df = pd.DataFrame([metrics])
                            st.dataframe(metrics_df)
                            
                            # Display forecast data
                            st.subheader("Forecast Data")
                            forecast = results['forecast']
                            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                            
                            # Only show future forecasts
                            last_historical_date = results['original_data']['ds'].max()
                            future_forecast = forecast_display[forecast_display['Date'] > last_historical_date]
                            st.dataframe(future_forecast)
                            
                            # Download links
                            st.subheader("Download Results")
                            col1, col2 = st.columns(2)
                            
                            # Read the CSV files to create download buttons
                            with open(results['forecast_path'], 'rb') as f:
                                forecast_csv = f.read()
                            
                            with open(results['compare_path'], 'rb') as f:
                                compare_csv = f.read()
                            
                            with col1:
                                st.download_button(
                                    label="Download Forecast Data",
                                    data=forecast_csv,
                                    file_name="forecast_results.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                st.download_button(
                                    label="Download Comparison Data",
                                    data=compare_csv,
                                    file_name="forecast_compare_results.csv",
                                    mime="text/csv"
                                )
                            
                        except Exception as e:
                            st.error(f"Error generating forecast: {str(e)}")
                else:
                    st.error("Please map your columns to 'ds' and 'y' before generating the forecast.")
    
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")

else:
    # Show instructions when no file is uploaded
    st.info("Please upload a CSV file using the sidebar to get started.")
    
    st.markdown("""
    ## Instructions
    
    ### Data Format Requirements
    1. Your CSV file should contain at least two columns:
       - A date column (will be mapped to 'ds')
       - A value column (will be mapped to 'y')
    2. The date column should be in a standard date format (e.g., YYYY-MM-DD, DD/MM/YYYY)
    3. The value column should contain numeric values to forecast
    
    ### Parameter Settings
    - **Number of periods**: How many future periods to forecast
    - **Time Granularity**: Choose daily, weekly, or monthly forecasting
    - **Advanced Parameters**: Fine-tune the forecasting model
      - IQR for outlier detection: Uses the Interquartile Range method to identify and handle outliers
      - Log Transform: Helps with data that has exponential growth or high variance
      - Cross-Validation: Tests the model's accuracy on historical data
    
    ### Output
    - Forecast comparison plot showing actual vs. predicted values
    - Forecast components showing trends and seasonality
    - Metrics evaluating forecast accuracy
    - Downloadable forecast data
    """)

