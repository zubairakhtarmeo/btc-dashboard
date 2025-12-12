"""
Configuration Management
========================

Centralized configuration for the crypto prediction system
"""

import os
from typing import Dict, Any
import json


class Config:
    """Main configuration class"""
    
    # API Keys (set via environment variables or config file)
    API_KEYS = {
        'newsapi': os.getenv('NEWSAPI_KEY', ''),
        'twitter': os.getenv('TWITTER_API_KEY', ''),
        'reddit': os.getenv('REDDIT_API_KEY', ''),
        'glassnode': os.getenv('GLASSNODE_API_KEY', ''),
        'coingecko': os.getenv('COINGECKO_API_KEY', ''),
        'binance': os.getenv('BINANCE_API_KEY', ''),
        'cryptopanic': os.getenv('CRYPTOPANIC_API_KEY', ''),
        'opensea': os.getenv('OPENSEA_API_KEY', '')  # NFT data
    }
    
    # Data Collection Settings
    DATA_COLLECTION = {
        'default_symbol': 'bitcoin',
        'update_interval_minutes': 60,  # How often to fetch new data
        'historical_hours': 720,  # 30 days
        'price_interval': '1h',
        'max_news_articles': 100,
        'max_social_posts': 1000
    }
    
    # Model Settings
    MODEL = {
        'sequence_length': 60,  # Hours of history to consider
        'prediction_horizons': [1, 6, 24],  # Multi-horizon prediction (hours)
        'n_features': 50,  # Will be auto-adjusted
        'architecture': 'hybrid',  # 'lstm', 'gru', 'tcn', 'hybrid'
        'lstm_units': [128, 64],
        'gru_units': [128, 64],
        'tcn_filters': [64, 64],
        'attention_heads': 4,
        'attention_key_dim': 32,
        'dense_units': [64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2,
        'early_stopping_patience': 15,
        'use_mixed_precision': True,  # Faster training on GPU
        'mc_dropout_samples': 10  # For uncertainty estimation
    }
    
    # Ensemble Settings
    ENSEMBLE = {
        'use_ensemble': True,
        'model_architectures': ['lstm', 'gru', 'tcn', 'hybrid'],
        'use_time_series_cv': True,  # Walk-forward validation
        'n_cv_splits': 3
    }
    
    # Feature Engineering Settings
    FEATURES = {
        'technical_indicators': True,
        'sentiment_analysis': True,
        'onchain_metrics': True,
        'market_indicators': True,
        'lag_features': True,
        'rolling_windows': [7, 14, 30, 168]
    }
    
    # Sentiment Analysis Settings
    SENTIMENT = {
        'use_transformers': True,  # Use advanced models (slower but more accurate)
        'news_weight': 0.6,
        'social_weight': 0.4,
        'sentiment_window_hours': 24,
        'anomaly_threshold': 2.0  # Z-score threshold
    }
    
    # Signal Generation Settings
    SIGNALS = {
        'risk_tolerance': 'medium',  # low, medium, high
        'min_confidence': 0.65,
        'use_stop_loss': True,
        'position_size_limits': {
            'low': 0.05,
            'medium': 0.10,
            'high': 0.20
        }
    }
    
    # Logging Settings
    LOGGING = {
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'crypto_predictor.log',
        'max_bytes': 10485760,  # 10MB
        'backup_count': 5
    }
    
    # Paths
    PATHS = {
        'models': 'models/',
        'data': 'data/',
        'logs': 'logs/',
        'cache': 'cache/'
    }
    
    # Performance Settings
    PERFORMANCE = {
        'use_gpu': True,
        'num_workers': 4,
        'cache_data': True,
        'parallel_data_collection': True
    }
    
    # Backtesting Settings
    BACKTESTING = {
        'enabled': False,  # Set True to run backtest
        'initial_capital': 10000.0,
        'position_size': 0.1,
        'transaction_cost': 0.001,  # 0.1%
        'walk_forward_window': 720  # 30 days
    }
    
    # Regime Detection Settings
    REGIME_DETECTION = {
        'enabled': True,
        'lookback_window': 100,
        'volatility_window': 20,
        'regime_threshold': 0.15,
        'adapt_model_parameters': True,  # Adjust params based on regime
        'regime_history_length': 50
    }
    
    # Alternative Data Settings
    ALTERNATIVE_DATA = {
        'enabled': True,
        'google_trends': True,
        'nft_activity': False,  # Requires OpenSea API key
        'derivatives_data': True,
        'funding_rates': True,
        'liquidations': True,
        'options_flow': True,
        'collection_interval_minutes': 60
    }
    
    # Portfolio Optimization Settings
    PORTFOLIO = {
        'enabled': False,  # Enable for multi-asset mode
        'total_capital': 10000.0,
        'max_position_size': 0.2,  # 20% max per asset
        'risk_free_rate': 0.02,  # 2% annual
        'rebalance_threshold': 0.05,  # Rebalance if drift > 5%
        'optimization_method': 'sharpe',  # 'sharpe', 'min_variance', 'risk_parity', 'kelly'
        'min_assets': 3,
        'max_assets': 10,
        'correlation_lookback': 90  # days
    }
    
    # Explainability Settings
    EXPLAINABILITY = {
        'enabled': True,
        'use_shap': True,  # Requires shap library
        'extract_attention': True,
        'feature_importance_method': 'perturbation',  # 'perturbation' or 'gradient'
        'top_features_display': 10,
        'generate_visualizations': True,
        'save_explanations': True,
        'explanation_path': 'explanations/'
    }
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        for key, value in config_data.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def save_to_file(cls, filepath: str):
        """Save configuration to JSON file"""
        config_data = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                config_data[key] = getattr(cls, key)
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def get_api_key(cls, service: str) -> str:
        """Get API key for a service"""
        return cls.API_KEYS.get(service, '')


# Create default config file if it doesn't exist
if not os.path.exists('config.json'):
    Config.save_to_file('config.json')
