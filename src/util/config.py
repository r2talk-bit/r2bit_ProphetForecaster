#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for Prophet Forecaster.
Contains parameters that control the forecasting process.
"""

# Forecast period configuration
FORECAST_PERIOD = {
    'unit': 'months',  # Options: 'days', 'months'
    'periods': 30,   # Number of periods to forecast
}

# Configurações do Prophet
PROPHET_PARAMS = {
    'growth': 'linear',
    'seasonality_mode': 'multiplicative',  # Alterado para multiplicativo devido à alta volatilidade
    'daily_seasonality': False,           # Mantido False pois os dados são diários
    'weekly_seasonality': True,           # Mantido True para capturar padrões semanais
    'yearly_seasonality': True,           # Mantido True para capturar sazonalidade anual
    'changepoint_prior_scale': 0.05,      # Valor padrão, pode ser ajustado após validação
    'seasonality_prior_scale': 10.0,      # Valor padrão, pode ser ajustado após validação
    'holidays_prior_scale': 10.0          # Controla o impacto dos feriados nas previsões
}

# Configuração de regressores personalizados
# Estes são fatores externos que podem influenciar as vendas
# Exemplo: promoções, mudanças de preço, ações de marketing, etc.
REGRESSORS = {
    # Defina aqui os regressores que deseja usar
    # 'promo': {'active': False, 'mode': 'multiplicative', 'standardize': True},
    # 'preco': {'active': False, 'mode': 'multiplicative', 'standardize': True},
    # 'marketing': {'active': False, 'mode': 'additive', 'standardize': True}
}

# Date format configuration
DATE_FORMAT = '%Y-%m-%d'  # ISO format (YYYY-MM-DD)

# CSV configuration
CSV_FORMAT = {
    'separator': ',',     # Separador de colunas: ',' ou ';'
    'decimal': '.',       # Separador decimal: '.' ou ','
}

# Logging configuration
LOG_LEVEL = 'INFO'

# Default file paths
DEFAULT_INPUT_FILE = 'examples/input.csv'
DEFAULT_OUTPUT_FILE = 'outputs/forecast.csv'
DEFAULT_METRICS_FILE = 'outputs/forecast_metrics.csv'
