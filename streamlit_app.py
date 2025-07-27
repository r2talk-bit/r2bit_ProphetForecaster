import streamlit as st
import pandas as pd
import io
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add the project directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the forecaster function
from src.util.forecaster import doForecast, plot_forecast

st.set_page_config(page_title="Prophet Forecaster", layout="wide")

st.title("Prophet Forecaster")
st.write("Upload a CSV file with sales data to generate forecasts")

# File upload widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file with semicolon separator
        df = pd.read_csv(uploaded_file, sep=";")
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(df.head())
        
        # Check if the required columns exist
        if 'DATA' not in df.columns:
            st.error("The CSV file must contain a 'DATA' column in the format dd-mm-yyyy")
        else:
            # Check if there are any SKU columns
            sku_columns = [col for col in df.columns if col.startswith('QUANTIDADE_VENDIDA_SKU_')]
            
            if not sku_columns:
                st.error("The CSV file must contain at least one column with the name format 'QUANTIDADE_VENDIDA_SKU_XXX'")
            else:
                # Display information about the data
                st.write(f"Found {len(sku_columns)} SKU columns in the data")
                
                # Parameters for forecasting
                st.subheader("Forecast Parameters")
                periods = st.slider("Number of days to forecast", min_value=7, max_value=365, value=30, step=1)
                
                if st.button("Generate Forecasts"):
                    with st.spinner("Generating forecasts..."):
                        # Call the forecasting function
                        forecast_results = doForecast(df, periods=periods)
                        
                        # Display results
                        st.subheader("Forecast Results")
                        
                        # Create tabs for each SKU
                        tabs = st.tabs([f"SKU {sku_id}" for sku_id in forecast_results.keys()])
                        
                        for i, (sku_id, result) in enumerate(forecast_results.items()):
                            with tabs[i]:
                                if 'error' in result:
                                    st.error(f"Error forecasting SKU {sku_id}: {result['error']}")
                                else:
                                    # Display forecast plot
                                    try:
                                        fig = plot_forecast(forecast_results, sku_id)
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Display forecast data
                                        forecast_df = result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                                        forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                                        st.dataframe(forecast_df.tail(periods))
                                        
                                        # Download link for forecast data
                                        csv = forecast_df.to_csv(index=False)
                                        st.download_button(
                                            label=f"Download forecast data for SKU {sku_id}",
                                            data=csv,
                                            file_name=f"forecast_sku_{sku_id}.csv",
                                            mime="text/csv"
                                        )
                                    except Exception as e:
                                        st.error(f"Error displaying forecast for SKU {sku_id}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")

# Add some instructions at the bottom
st.markdown("""
### Instructions
1. Prepare your CSV file with semicolon (;) as the separator
2. Make sure it has a 'DATA' column with dates in dd-mm-yyyy format
3. Include one or more columns with names like 'QUANTIDADE_VENDIDA_SKU_123' where '123' is the SKU ID
4. Upload the file using the file uploader above
5. Adjust the forecast parameters if needed
6. Click 'Generate Forecasts' to see the results
""")
