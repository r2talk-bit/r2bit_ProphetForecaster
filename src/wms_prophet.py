#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WMS Prophet Forecaster

This script implements a forecasting solution for warehouse sales using Facebook Prophet.
It reads data from a CSV file, generates forecasts, and outputs evaluation metrics
and comparison graphs between real and forecasted values.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from dateutil import parser
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_arg_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='WMS Prophet Forecaster - Time series forecasting using Prophet'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='examples/input.csv',
        help='Path to input CSV file (default: examples/input.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Path to output directory (default: outputs)'
    )
    parser.add_argument(
        '--periods', '-p',
        type=int,
        default=30,
        help='Number of periods to forecast (default: 30)'
    )
    parser.add_argument(
        '--delimiter', '-d',
        type=str,
        default=None,
        help='CSV delimiter (default: auto-detect between "," and ";")'
    )
    
    # Time granularity option
    parser.add_argument(
        '--granularity', '-g',
        type=str,
        default='daily',
        choices=['daily', 'weekly', 'monthly'],
        help='Time granularity for forecasting (default: daily)'
    )
    
    # Outlier handling options
    parser.add_argument(
        '--use-iqr',
        action='store_true',
        help='Use IQR method for outlier detection and handling'
    )
    parser.add_argument(
        '--iqr-multiplier',
        type=float,
        default=1.5,
        help='Multiplier for IQR to determine outlier thresholds (default: 1.5)'
    )
    parser.add_argument(
        '--max-threshold',
        type=float,
        default=None,
        help='Maximum threshold for values. Values above this will be capped'
    )
    
    # Data transformation options
    parser.add_argument(
        '--log-transform',
        action='store_true',
        help='Apply log transformation to the data (helps with large variations)'
    )
    
    # Model evaluation options
    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform cross-validation to find optimal parameters'
    )
    
    return parser


def load_data(file_path, delimiter=None):
    """
    Load data from CSV file with auto-detection of delimiter.
    
    Args:
        file_path (str): Path to the CSV file
        delimiter (str, optional): CSV delimiter. If None, auto-detect between "," and ";"
        
    Returns:
        pd.DataFrame: DataFrame with the loaded data
    """
    try:
        # If delimiter is not specified, try to auto-detect
        if delimiter is None:
            # Try comma first
            try:
                df = pd.read_csv(file_path, delimiter=',')
                if len(df.columns) > 1:
                    logger.info(f"Detected delimiter: ','")
                    return df
            except Exception:
                pass
            
            # Try semicolon
            try:
                df = pd.read_csv(file_path, delimiter=';')
                if len(df.columns) > 1:
                    logger.info(f"Detected delimiter: ';'")
                    return df
            except Exception:
                pass
            
            # If both failed, raise an error
            raise ValueError("Could not auto-detect delimiter. Please specify using --delimiter")
        else:
            # Use the specified delimiter
            df = pd.read_csv(file_path, delimiter=delimiter)
            return df
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)


def validate_data(df, max_threshold=None, use_iqr=True, iqr_multiplier=1.5, log_transform=False, granularity='daily'):
    """
    Validate the input data format and handle outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        max_threshold (float, optional): Maximum threshold for values in 'y' column.
                                       Values above this threshold will be capped.
        use_iqr (bool, optional): Whether to use IQR method for outlier detection.
                                 Default is True.
        iqr_multiplier (float, optional): Multiplier for IQR to determine outlier threshold.
                                        Default is 1.5 (standard for outlier detection).
        log_transform (bool, optional): Whether to apply log transformation to the data.
                                      Helps with large variations in the data.
        granularity (str, optional): Time granularity for forecasting ('daily', 'weekly', or 'monthly').
                                   Default is 'daily'.
        
    Returns:
        pd.DataFrame: Validated and processed DataFrame
    """
    # Check required columns
    required_columns = ['ds', 'y']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Detect and convert date column to datetime in YYYY-MM-DD format
    try:
        # First check if we need to convert at all (sample a few values)
        sample_dates = df['ds'].head(5).tolist()
        date_formats = []
        
        # Try to detect the format from the first few rows
        for date_str in sample_dates:
            if isinstance(date_str, str):
                # Check if it's already in YYYY-MM-DD format
                if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    date_formats.append('YYYY-MM-DD')
                # Check if it's in DD/MM/YYYY format
                elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
                    date_formats.append('DD/MM/YYYY')
                # Check if it's in MM/DD/YYYY format
                elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
                    date_formats.append('MM/DD/YYYY')
                # Other formats can be added here
        
        # If we detected DD/MM/YYYY format
        if 'DD/MM/YYYY' in date_formats:
            logger.info("Detected DD/MM/YYYY date format, converting to YYYY-MM-DD")
            df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        # If we detected MM/DD/YYYY format
        elif 'MM/DD/YYYY' in date_formats:
            logger.info("Detected MM/DD/YYYY date format, converting to YYYY-MM-DD")
            df['ds'] = pd.to_datetime(df['ds'], format='%m/%d/%Y')
        else:
            # Use dateutil parser which can handle various formats automatically
            logger.info("Using automatic date format detection")
            df['ds'] = df['ds'].apply(lambda x: parser.parse(str(x)) if isinstance(x, str) else x)
            df['ds'] = pd.to_datetime(df['ds'])
            
        # Ensure the date is in the correct format for Prophet
        df['ds'] = pd.to_datetime(df['ds']).dt.strftime('%Y-%m-%d')
        df['ds'] = pd.to_datetime(df['ds'])
        logger.info(f"Date range: {df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        logger.error(f"Error converting date column: {str(e)}")
        sys.exit(1)
    
    # Sort by date
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Apply log transformation if specified (but generally not recommended for this dataset)
    if log_transform:
        # Save original values before transformation
        df['original_y'] = df['y'].copy()
        
        # Apply log transformation (add small constant to handle zeros)
        # Only apply to non-zero values to preserve zeros
        mask = df['y'] > 0
        df.loc[mask, 'y'] = np.log1p(df.loc[mask, 'y'])  # log(1+y) to handle small values
        logger.info("Applied log transformation to non-zero data points")
    
    # Resample to weekly or monthly if requested - do this before outlier detection
    if granularity == 'weekly':
        logger.info("Resampling data to weekly granularity")
        # Set the date as index for resampling
        df = df.set_index('ds')
        # Resample to weekly frequency (week starts on Monday)
        df = df.resample('W-MON').sum().reset_index()
        logger.info(f"After resampling: {len(df)} data points")
    elif granularity == 'monthly':
        logger.info("Resampling data to monthly granularity")
        # Set the date as index for resampling
        df = df.set_index('ds')
        # Resample to monthly frequency (month starts on the 1st)
        df = df.resample('MS').sum().reset_index()
        logger.info(f"After resampling: {len(df)} data points")
    
    # Handle outliers using IQR method
    if use_iqr:
        df = handle_outliers_iqr(df, iqr_multiplier, max_threshold)
    
    # Handle outliers if max_threshold is provided (as a secondary option)
    elif max_threshold is not None:
        # Count outliers before capping
        outlier_count = (df['y'] > max_threshold).sum()
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers with values > {max_threshold}")
            # Cap values at max_threshold
            df['y'] = df['y'].clip(upper=max_threshold)
            logger.info(f"Capped outlier values at {max_threshold}")
    
    # Check if there's enough data
    min_required_points = 14  # At least 2 weeks of data
    if len(df) < min_required_points:
        logger.warning(f"Not enough data points: {len(df)}. Minimum required: {min_required_points}")
        logger.warning("Forecast may not be reliable")
        
    return df


def handle_outliers_iqr(df, iqr_multiplier, max_threshold=None):
    """
    Handle outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        iqr_multiplier (float): Multiplier for IQR to determine outlier threshold
        max_threshold (float, optional): Maximum threshold for values in 'y' column.
                                       Values above this threshold will be capped.
        
    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    # Calculate IQR for the 'y' column (only for non-zero values)
    non_zero_values = df['y'][df['y'] > 0]
    Q1 = non_zero_values.quantile(0.25)
    Q3 = non_zero_values.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate bounds
    lower_bound = max(0, Q1 - (iqr_multiplier * IQR))  # Ensure lower bound is not negative
    upper_bound = Q3 + (iqr_multiplier * IQR)
    
    # Apply max threshold if specified (take the minimum of IQR upper bound and max_threshold)
    if max_threshold is not None:
        upper_bound = min(upper_bound, max_threshold)
        logger.info(f"Using max threshold: {max_threshold}")
    
    # Identify outliers (but don't consider zeros as outliers)
    outliers = df[(df['y'] > 0) & ((df['y'] < lower_bound) | (df['y'] > upper_bound))]
    num_outliers = len(outliers)
    
    logger.info(f"IQR method identified {num_outliers} outliers")
    logger.info(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    logger.info(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    
    # Cap values outside the bounds (but preserve zeros)
    mask = df['y'] > 0  # Only modify non-zero values
    df.loc[mask, 'y'] = df.loc[mask, 'y'].clip(lower=lower_bound, upper=upper_bound)
    logger.info(f"Capped outlier values between {lower_bound:.2f} and {upper_bound:.2f} (zeros preserved)")
    
    # Log some sample outliers
    for i, (idx, row) in enumerate(outliers.iterrows()):
        if i < 5:  # Show only first 5 outliers
            original_value = row['y']
            capped_value = min(max(original_value, lower_bound), upper_bound)
            logger.info(f"Outlier at {row['ds'].strftime('%Y-%m-%d')}: {original_value:.2f} (capped to {capped_value:.2f})")
        else:
            break
    
    return df


def train_prophet_model(df, periods=30, log_transform=False, granularity='daily'):
    """
    Train a Prophet model and generate forecasts.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'ds' and 'y' columns
        periods (int): Number of periods to forecast
        log_transform (bool): Whether log transformation was applied to the data
        granularity (str): Time granularity for forecasting ('daily', 'weekly', or 'monthly')
        
    Returns:
        tuple: (forecast_df, model)
    """
    logger.info(f"Training Prophet model with {len(df)} data points")
    
    # Configure model with improved seasonality parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # We'll add a custom weekly seasonality below
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for data with zeros
        seasonality_prior_scale=20,  # Adjusted for better fit
        changepoint_prior_scale=0.5,  # Adjusted for better trend detection
        interval_width=0.95,
        changepoint_range=0.8,  # Allow changepoints throughout most of the history
        uncertainty_samples=1000  # Increase number of samples for better uncertainty estimation
    )
    
    # Add Brazilian holidays
    model.add_country_holidays(country_name='BR')
    
    # Add appropriate seasonality components based on granularity
    if granularity == 'daily':
        # For daily data, add custom weekly seasonality with higher order Fourier terms
        model.add_seasonality(
            name='weekly',
            period=7,
            fourier_order=5,  # Higher order for more flexible weekly patterns
            prior_scale=10
        )
        logger.info("Added weekly seasonality component for daily granularity")
    elif granularity == 'weekly':
        # For weekly data, use custom yearly seasonality based on weeks
        # Disable default yearly seasonality first
        model = Prophet(yearly_seasonality=False,
                       weekly_seasonality=False,
                       daily_seasonality=False,
                       seasonality_mode='multiplicative',
                       seasonality_prior_scale=20,
                       changepoint_prior_scale=0.5,
                       interval_width=0.80,  # Changed to 80% confidence interval
                       changepoint_range=0.8,
                       uncertainty_samples=1000)
        
        # Add Brazilian holidays
        model.add_country_holidays(country_name='BR')
        
        # Add custom yearly seasonality based on weeks
        model.add_seasonality(
            name='yearly',
            period=52.18,  # 365.25/7 weeks per year
            fourier_order=6,  # Reduced to avoid overfitting
            prior_scale=10
        )
        logger.info("Added custom yearly seasonality for weekly granularity (period=52.18 weeks)")
    elif granularity == 'monthly':
        # For monthly data, focus on quarterly patterns
        model.add_seasonality(
            name='quarterly',
            period=3,  # 3 months per quarter
            fourier_order=2,
            prior_scale=10
        )
        logger.info("Added quarterly seasonality component for monthly granularity")
    
    # Add weekend regressor only for daily granularity
    if granularity == 'daily':
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        model.add_regressor('is_weekend', prior_scale=25, standardize=False)  # Increased prior scale for stronger weekend effect
        logger.info("Added weekend regressor for daily granularity")
    
    # Fit model first to generate holidays
    model.fit(df)
    
    # Add holiday regressor based on the fitted model's holidays
    df['is_holiday'] = 0  # Initialize with zeros
    
    # Mark Brazilian holidays as 1 in the is_holiday column
    if hasattr(model, 'holidays') and model.holidays is not None:
        for holiday_name, dates in model.holidays.items():
            for holiday_date in dates:
                holiday_indices = df[df['ds'] == holiday_date].index
                if len(holiday_indices) > 0:
                    df.loc[holiday_indices, 'is_holiday'] = 1
    
    # Log data summary for debugging
    logger.info(f"Data summary after holiday marking: min={df['y'].min()}, max={df['y'].max()}, mean={df['y'].mean():.2f}")
    
    # Create a new model with the same parameters but now including the holiday regressor
    # For daily and monthly data
    if granularity != 'weekly':  # Only create the model here for daily and monthly
        model_with_holiday = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # We'll add custom seasonality based on granularity below
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=20,  # Adjusted for better fit
            changepoint_prior_scale=0.5,
            interval_width=0.95,
            changepoint_range=0.8,
            uncertainty_samples=1000
        )
        
        # Add the same components to the new model based on granularity
        model_with_holiday.add_country_holidays(country_name='BR')
    # For weekly data, the model will be created in the seasonality section below
    
    # Add appropriate seasonality components based on granularity
    if granularity == 'daily':
        model_with_holiday.add_seasonality(
            name='weekly',
            period=7,
            fourier_order=5,
            prior_scale=10
        )
        model_with_holiday.add_regressor('is_weekend', prior_scale=25, standardize=False)
        logger.info("Added weekly seasonality and weekend regressor to holiday model for daily granularity")
    elif granularity == 'weekly':
        # For weekly data, use custom yearly seasonality based on weeks
        # Recreate model_with_holiday with yearly_seasonality=False
        model_with_holiday = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',  # More stable than multiplicative for weekly data
            seasonality_prior_scale=10,   # Moderate strength
            changepoint_prior_scale=0.05, # More conservative to prevent overfitting
            interval_width=0.80,          # Tighter prediction bounds
            changepoint_range=0.8,
            uncertainty_samples=1000
        )
        
        # Add Brazilian holidays
        model_with_holiday.add_country_holidays(country_name='BR')
        
        # Add custom yearly seasonality based on weeks
        model_with_holiday.add_seasonality(
            name='yearly',
            period=52.18,  # 365.25/7 weeks per year
            fourier_order=6,  # Reduced to avoid overfitting
            prior_scale=10
        )
        logger.info("Added custom yearly seasonality to holiday model for weekly granularity (period=52.18 weeks)")
        
        # Add the holiday regressor
        model_with_holiday.add_regressor('is_holiday', prior_scale=15, standardize=False, mode='multiplicative')
        logger.info("Added holiday regressor to model for weekly granularity")
    elif granularity == 'monthly':
        model_with_holiday.add_seasonality(
            name='quarterly',
            period=3,  # 3 months per quarter
            fourier_order=2,
            prior_scale=10
        )
        logger.info("Added quarterly seasonality to holiday model for monthly granularity")
    
    # Add holiday regressor for daily and monthly granularity
    if granularity != 'weekly':  # Skip for weekly as it's already added above
        model_with_holiday.add_regressor('is_holiday', prior_scale=15, standardize=False, mode='multiplicative')
        logger.info(f"Added holiday regressor to model for {granularity} granularity")
    
    # Fit the new model
    model_with_holiday.fit(df)
    
    # Use the new model from now on
    model = model_with_holiday
    
    # Create future dataframe
    if granularity == 'weekly':
        # For weekly granularity, we need to adjust the periods
        # Each period is one week instead of one day
        future = model.make_future_dataframe(periods=periods, freq='W')
        logger.info(f"Generating {periods} weeks of forecasts")
    elif granularity == 'monthly':
        # For monthly granularity, each period is one month
        future = model.make_future_dataframe(periods=periods, freq='MS')
        logger.info(f"Generating {periods} months of forecasts")
    else:
        future = model.make_future_dataframe(periods=periods)
        logger.info(f"Generating {periods} days of forecasts")
    
    # Add weekend regressor to future dataframe only for daily granularity
    if 'is_weekend' in df.columns and granularity == 'daily':
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        logger.info("Added weekend regressor to future dataframe for daily granularity")
    
    # Add holiday regressor to future dataframe if it was used in training
    if 'is_holiday' in df.columns:
        future['is_holiday'] = 0  # Initialize with zeros
        
        # Mark Brazilian holidays in future dataframe
        if hasattr(model, 'holidays') and model.holidays is not None:
            for holiday_name, dates in model.holidays.items():
                for holiday_date in dates:
                    holiday_indices = future[future['ds'] == holiday_date].index
                    if len(holiday_indices) > 0:
                        future.loc[holiday_indices, 'is_holiday'] = 1
    
    # Generate forecast
    forecast = model.predict(future)
    
    # If log transformation was applied, reverse it
    if log_transform:
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[col] = np.expm1(forecast[col])  # exp(y)-1 is inverse of log1p
        logger.info("Reversed log transformation in forecast")
    
    # Ensure non-negative values
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
    
    # Calculate historical non-zero minimum and average for setting floor values
    historical_nonzero = df[df['y'] > 0]['y']
    if len(historical_nonzero) > 0:
        historical_min = historical_nonzero.min()
        historical_mean = historical_nonzero.mean()
        historical_median = historical_nonzero.median()
        
        # Calculate a reasonable minimum forecast value (25% of median for non-zero values)
        min_forecast_value = historical_median * 0.25
        
        # Get the forecast start date (first date after historical data ends)
        max_historical_date = df['ds'].max()
        future_mask = forecast['ds'] > max_historical_date
        
        # For future forecasts only, ensure we don't have unreasonably low values
        if future_mask.any():
            # Replace very low future forecasts with the minimum forecast value
            low_future_forecasts = (forecast['yhat'] < min_forecast_value) & future_mask
            if low_future_forecasts.any():
                forecast.loc[low_future_forecasts, 'yhat'] = min_forecast_value
                # Adjust bounds accordingly
                forecast.loc[low_future_forecasts, 'yhat_lower'] = min_forecast_value * 0.5
                forecast.loc[low_future_forecasts, 'yhat_upper'] = min_forecast_value * 2.0
                
                logger.info(f"Adjusted {low_future_forecasts.sum()} future forecasts below minimum threshold of {min_forecast_value:.2f}")
        
        logger.info(f"Historical data stats - min: {historical_min:.2f}, mean: {historical_mean:.2f}, median: {historical_median:.2f}")
    
    # Widen confidence intervals to improve coverage
    # Calculate the average forecast value
    avg_forecast = forecast['yhat'].mean()
    avg_y = df['y'].mean()
    
    # Calculate standard deviation of actual values for better interval scaling
    y_std = df['y'].std()
    
    # Dynamically adjust confidence intervals based on data characteristics
    if granularity == 'daily':
        width_factor = 2.0  # Wider for daily data (more volatile)
    elif granularity == 'weekly':
        width_factor = 2.5  # Even wider for weekly data
    else:  # monthly
        width_factor = 3.0  # Widest for monthly data (fewer data points)
    
    # Apply dynamic width adjustment
    forecast['yhat_lower'] = forecast['yhat'] - (forecast['yhat'] - forecast['yhat_lower']) * width_factor
    forecast['yhat_upper'] = forecast['yhat'] + (forecast['yhat_upper'] - forecast['yhat']) * width_factor
    
    # For very low values, ensure minimum width based on data standard deviation
    min_width = y_std * 0.5
    narrow_intervals = (forecast['yhat_upper'] - forecast['yhat_lower']) < min_width
    if narrow_intervals.any():
        forecast.loc[narrow_intervals, 'yhat_lower'] = forecast.loc[narrow_intervals, 'yhat'] - min_width/2
        forecast.loc[narrow_intervals, 'yhat_upper'] = forecast.loc[narrow_intervals, 'yhat'] + min_width/2
    
    # Ensure lower bounds don't go below zero
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    
    logger.info(f"Adjusted confidence intervals with width factor {width_factor} for {granularity} granularity")
    logger.info(f"Average interval width: {(forecast['yhat_upper'] - forecast['yhat_lower']).mean():.2f}")
    
    return forecast, model


def cross_validate_model(df, initial='90 days', period='30 days', horizon='60 days'):
    """
    Perform cross-validation on the Prophet model.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'ds' and 'y' columns
        initial (str): Initial training period
        period (str): Period between cutoff dates
        horizon (str): Forecast horizon
        
    Returns:
        pd.DataFrame: Cross-validation metrics
    """
    logger.info(f"Performing cross-validation with initial={initial}, period={period}, horizon={horizon}")
    
    try:
        # Create a simple model for cross-validation
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            uncertainty_samples=1000  # Increase number of samples for better uncertainty estimation
        )
        
        # Fit the model first
        model.fit(df)
        
        # Perform cross-validation
        df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
        
        # Calculate metrics
        df_metrics = performance_metrics(df_cv)
        
        logger.info(f"Cross-validation metrics: MAE={df_metrics['mae'].mean():.2f}, MAPE={df_metrics['mape'].mean():.2f}%")
        
        return df_metrics
    except Exception as e:
        logger.warning(f"Cross-validation failed: {str(e)}. Continuing without cross-validation.")
        return None


def calculate_metrics(df, forecast):
    """
    Calculate forecast evaluation metrics.
    
    Args:
        df (pd.DataFrame): Original data
        forecast (pd.DataFrame): Forecast data
        
    Returns:
        dict: Dictionary with metrics
    """
    # Merge actual values with forecasts
    evaluation_df = pd.merge(
        df[['ds', 'y']],
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='left'
    )
    
    # Calculate metrics
    metrics = {}
    
    # Mean Absolute Error
    metrics['MAE'] = np.mean(np.abs(evaluation_df['y'] - evaluation_df['yhat']))
    
    # Root Mean Square Error
    metrics['RMSE'] = np.sqrt(np.mean((evaluation_df['y'] - evaluation_df['yhat'])**2))
    
    # Mean Absolute Percentage Error (avoiding division by zero)
    non_zero_mask = evaluation_df['y'] > 0
    if non_zero_mask.any():
        metrics['MAPE'] = np.mean(
            np.abs((evaluation_df.loc[non_zero_mask, 'y'] - 
                   evaluation_df.loc[non_zero_mask, 'yhat']) / 
                   evaluation_df.loc[non_zero_mask, 'y'])
        ) * 100
    else:
        metrics['MAPE'] = np.nan
    
    # Coverage (percentage of actual values within prediction interval)
    within_interval = ((evaluation_df['y'] >= evaluation_df['yhat_lower']) & 
                       (evaluation_df['y'] <= evaluation_df['yhat_upper']))
    metrics['Coverage'] = within_interval.mean() * 100
    
    # Log coverage details for debugging
    total_points = len(evaluation_df.dropna())
    points_within = within_interval.sum()
    logger.info(f"Coverage: {points_within}/{total_points} points ({metrics['Coverage']:.2f}%) within prediction interval")
    
    return metrics


def generate_comparison_plot(df, forecast, output_path):
    """
    Generate a comparison plot between actual and forecasted values.
    
    Args:
        df (pd.DataFrame): Original data
        forecast (pd.DataFrame): Forecast data
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.plot(df['ds'], df['y'], 'o', markersize=4, color='#1f77b4', label='Actual')
    
    # Get the last date from historical data
    last_date = df['ds'].max()
    
    # Split forecast into historical and future
    historical_forecast = forecast[forecast['ds'] <= last_date]
    future_forecast = forecast[forecast['ds'] > last_date]
    
    # Plot historical forecast
    plt.plot(historical_forecast['ds'], historical_forecast['yhat'], 
             '-', color='#ff7f0e', label='Historical Forecast')
    
    # Plot future forecast
    plt.plot(future_forecast['ds'], future_forecast['yhat'], 
             '-', color='#2ca02c', label='Future Forecast')
    
    # Plot confidence intervals for future forecast
    plt.fill_between(
        future_forecast['ds'],
        future_forecast['yhat_lower'],
        future_forecast['yhat_upper'],
        color='#2ca02c',
        alpha=0.2,
        label='80% Confidence Interval'
    )
    
    # Add vertical line at the end of historical data
    plt.axvline(x=last_date, color='red', linestyle='--', 
                label=f'Forecast Start ({last_date.strftime("%Y-%m-%d")})')
    
    # Format the plot
    plt.title('Prophet Forecast: Actual vs Predicted', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Comparison plot saved to {output_path}")
    plt.close()


def save_metrics(metrics, output_path):
    """
    Save metrics to a CSV file.
    
    Args:
        metrics (dict): Dictionary with metrics
        output_path (str): Path to save the metrics
    """
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Metrics saved to {output_path}")


def save_forecast(forecast, df, output_path, compare_output_path=None):
    """
    Save forecast to a CSV file.
    
    Args:
        forecast (pd.DataFrame): Forecast DataFrame
        df (pd.DataFrame): Original data DataFrame with real values
        output_path (str): Path to save the forecast
        compare_output_path (str, optional): Path to save the comparison results
    """
    # Select relevant columns and rename for clarity
    forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_output.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
    
    # Format date column
    forecast_output['date'] = forecast_output['date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    forecast_output.to_csv(output_path, index=False)
    
    # Generate comparison results if path is provided
    if compare_output_path:
        # Create a copy of the forecast dataframe with only the dates and forecasted values
        compare_df = forecast[['ds', 'yhat']].copy()
        
        # Round the forecasted values to integers and handle NaN values
        compare_df['yhat'] = compare_df['yhat'].fillna(0).round().astype(int)
        
        # Merge with the original dataframe to get the real values
        compare_df = pd.merge(compare_df, df[['ds', 'y']], on='ds', how='left')
        
        # Round the real values to integers and handle NaN values
        compare_df['y'] = compare_df['y'].fillna(0).round().astype(int)
        
        # Calculate the percentage difference
        # Avoid division by zero by replacing zeros with NaN and then back to zero
        compare_df['y_non_zero'] = compare_df['y'].replace(0, np.nan)
        compare_df['pct_diff'] = ((compare_df['yhat'] - compare_df['y_non_zero']) / compare_df['y_non_zero'] * 100).round(2)
        compare_df['pct_diff'] = compare_df['pct_diff'].fillna(0)  # Replace NaN with 0 for cases where real value was 0
        
        # Format the date column
        compare_df['ds'] = compare_df['ds'].dt.strftime('%Y-%m-%d')
        
        # Select and rename columns for the final output
        compare_output = compare_df[['ds', 'y', 'yhat', 'pct_diff']].copy()
        compare_output.columns = ['date', 'real_value', 'forecast_value', 'pct_difference']
        
        # Save to CSV with semicolon separator
        compare_output.to_csv(compare_output_path, index=False, sep=';')
        logger.info(f"Forecast comparison results saved to {compare_output_path}")
        
    logger.info(f"Forecast saved to {output_path}")


def main():
    """Main function to run the forecasting process."""
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("Starting WMS Prophet Forecaster")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Forecast periods: {args.periods}")
    logger.info(f"Time granularity: {args.granularity}")
    
    # Load data
    df = load_data(args.input, args.delimiter)
    logger.info(f"Loaded {len(df)} data points")
    
    # Validate data and handle outliers
    df = validate_data(
        df,
        max_threshold=args.max_threshold,
        use_iqr=args.use_iqr,
        iqr_multiplier=args.iqr_multiplier,
        log_transform=args.log_transform,
        granularity=args.granularity
    )
    
    # Optionally perform cross-validation to find optimal parameters
    if args.cross_validate:
        cv_metrics = cross_validate_model(df, initial='90 days', period='30 days', horizon='60 days')
        if cv_metrics is not None:
            logger.info(f"Cross-validation complete. Use these metrics to optimize your model parameters.")
        else:
            logger.warning("Skipping cross-validation due to errors. Continuing with model training.")
    
    # Train model and generate forecast
    forecast, model = train_prophet_model(
        df, 
        periods=args.periods, 
        log_transform=args.log_transform,
        granularity=args.granularity
    )
    
    # Calculate metrics
    metrics = calculate_metrics(df, forecast)
    logger.info(f"Forecast metrics: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
    
    # Generate comparison plot
    plot_path = os.path.join(args.output, 'forecast_comparison.png')
    generate_comparison_plot(df, forecast, plot_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'forecast_metrics.csv')
    save_metrics(metrics, metrics_path)
    
    # Save forecast
    forecast_path = os.path.join(args.output, 'forecast_results.csv')
    compare_path = os.path.join(args.output, 'forecast_compare_results.csv')
    save_forecast(forecast, df, forecast_path, compare_path)
    logger.info(f"Forecast comparison results saved to {compare_path}")
    
    # Generate components plot
    components_path = os.path.join(args.output, 'forecast_components.png')
    fig = model.plot_components(forecast)
    fig.savefig(components_path)
    logger.info(f"Components plot saved to {components_path}")
    plt.close(fig)
    
    logger.info("Forecasting process completed successfully")


if __name__ == "__main__":
    main()
