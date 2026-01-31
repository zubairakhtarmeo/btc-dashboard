"""
Feature Engineering Module
==========================

Transforms raw data into ML-ready features:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price patterns and trends
- Volume analysis
- Sentiment features
- On-chain metrics
- Market factors
- Time-based features
- Cross-asset correlations

Author: AI/ML Trading System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.signal import find_peaks
import pickle
import os

from simple_features import add_simple_features

logger = logging.getLogger(__name__)


def _add_simple_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set used by train_simple/dashboard (kept here so feature_engineering
    can refresh the exact features expected by the current simplified model).
    """
    df = df.copy()

    # Basic returns
    for h in [1, 6, 12, 24, 48, 168]:
        df[f'return_{h}h'] = df['close'].pct_change(h)
        df[f'log_return_{h}h'] = np.log1p(df[f'return_{h}h'])

    # Simple moving averages
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

    # Volatility
    for period in [7, 14, 21, 50]:
        df[f'volatility_{period}'] = df['return_1h'].rolling(period).std()

    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']

    # Momentum
    df['momentum_consistency'] = (
        np.sign(df['return_1h']) == np.sign(df['return_6h'])
    ).astype(int)

    return df.dropna()


class FeatureEngineer:
    """
    Comprehensive feature engineering for cryptocurrency prediction
    with caching, normalization, and advanced features
    """
    
    def __init__(self, cache_dir: str = 'cache'):
        self.feature_names = []
        self.scalers = {}
        self.feature_metadata = {}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("FeatureEngineer initialized")
    
    def save_scalers(self, filepath: str = None):
        """Save scalers for reproducible inference"""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'feature_scalers.pkl')
        
        save_data = {
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'feature_metadata': self.feature_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str = None):
        """Load scalers for inference"""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'feature_scalers.pkl')
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.scalers = save_data['scalers']
            self.feature_names = save_data['feature_names']
            self.feature_metadata = save_data.get('feature_metadata', {})
            logger.info(f"Scalers loaded from {filepath}")
        else:
            logger.warning(f"Scaler file not found: {filepath}")
    
    def engineer_all_features(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame = None,
        social_df: pd.DataFrame = None,
        onchain_df: pd.DataFrame = None,
        market_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create all features from multi-source data
        
        Args:
            price_df: Price OHLCV data
            news_df: News with sentiment
            social_df: Social media data
            onchain_df: On-chain metrics
            market_df: Market indicators
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Start with price data
        features_df = price_df.copy()
        
        # 1. Technical indicators
        features_df = self.add_technical_indicators(features_df)
        logger.info("✓ Technical indicators added")
        
        # 2. Price patterns
        features_df = self.add_price_patterns(features_df)
        logger.info("✓ Price patterns added")
        
        # 3. Volume features
        features_df = self.add_volume_features(features_df)
        logger.info("✓ Volume features added")
        
        # 4. Time-based features
        features_df = self.add_time_features(features_df)
        logger.info("✓ Time features added")
        
        # 5. Sentiment features
        if news_df is not None or social_df is not None:
            features_df = self.add_sentiment_features(features_df, news_df, social_df)
            logger.info("✓ Sentiment features added")
        
        # 6. On-chain features
        if onchain_df is not None:
            features_df = self.add_onchain_features(features_df, onchain_df)
            logger.info("✓ On-chain features added")
        
        # 7. Market features
        if market_df is not None:
            features_df = self.add_market_features(features_df, market_df)
            logger.info("✓ Market features added")
        
        # 8. Lag features
        features_df = self.add_lag_features(features_df)
        logger.info("✓ Lag features added")
        
        # 9. Rolling statistics
        features_df = self.add_rolling_statistics(features_df)
        logger.info("✓ Rolling statistics added")
        
        # 10. Volatility-adjusted features
        features_df = self.add_volatility_adjusted_features(features_df)
        logger.info("✓ Volatility-adjusted features added")
        
        # 11. Dynamic sentiment velocity
        if news_df is not None or social_df is not None:
            features_df = self.add_sentiment_velocity(features_df)
            logger.info("✓ Sentiment velocity added")
        
        # 12. Normalized on-chain growth metrics
        if onchain_df is not None:
            features_df = self.add_normalized_onchain_growth(features_df)
            logger.info("✓ Normalized on-chain growth added")
        
        # 13. Multi-horizon momentum features
        features_df = self.add_multi_horizon_momentum(features_df)
        logger.info("✓ Multi-horizon momentum added")
        logger.info("✓ Rolling statistics added")
        
        # Remove NaN rows (from rolling calculations)
        initial_len = len(features_df)
        features_df = features_df.dropna()
        logger.info(f"Removed {initial_len - len(features_df)} rows with NaN values")
        
        self.feature_names = list(features_df.columns)
        logger.info(f"Feature engineering complete! Created {len(self.feature_names)} features")
        
        return features_df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        
        # 1. Moving Averages
        for period in [7, 14, 21, 30, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # 2. RSI (Relative Strength Index)
        for period in [14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 4. Bollinger Bands with %B
        for period in [20, 50]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            
            # Bollinger %B (position within bands)
            df[f'bb_percent_b_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
            )
            
            # Bollinger Band squeeze (low volatility)
            df[f'bb_squeeze_{period}'] = df[f'bb_width_{period}'].rolling(window=20).apply(
                lambda x: (x.iloc[-1] == x.min()).astype(float)
            )
        
        # 5. ATR (Average True Range) - Volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        df['atr_normalized'] = df['atr_14'] / df['close']
        
        # 6. Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'stoch_{period}_smooth'] = df[f'stoch_{period}'].rolling(window=3).mean()
        
        # 7. Momentum Indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100  # Rate of Change
        
        # 8. ADX (Average Directional Index) - Trend Strength
        df['adx'] = self._calculate_adx(df, period=14)
        
        # 9. OBV (On-Balance Volume)
        df['obv'] = self._calculate_obv(df)
        
        # 10. VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price patterns and trends"""
        
        # 1. Returns (different time horizons)
        for period in [1, 6, 12, 24, 48, 168]:  # 1h to 1 week
            df[f'return_{period}h'] = df['close'].pct_change(periods=period)
        
        # 2. Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 3. High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['open']
        
        # 4. Price position in daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 5. Upper/lower wicks (candlestick patterns)
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        
        # 6. Trend direction
        df['trend_7d'] = np.where(df['close'] > df['sma_168'], 1, -1) if 'sma_168' in df.columns else 0
        df['trend_30d'] = np.where(df['close'] > df['sma_720'], 1, -1) if 'sma_720' in df.columns else 0
        
        # 7. Price distance from moving averages
        for period in [7, 21, 50]:
            if f'sma_{period}' in df.columns:
                df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # 8. Volatility
        for window in [7, 14, 30]:
            df[f'volatility_{window}d'] = df['log_return'].rolling(window=window).std()
        
        # 9. Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        
        # 1. Volume moving averages
        for period in [7, 14, 30]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # 2. Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_14']
        
        # 3. Volume trend
        df['volume_trend'] = df['volume'].pct_change(periods=7)
        
        # 4. Price-Volume correlation
        for window in [7, 14, 30]:
            df[f'price_volume_corr_{window}'] = df['close'].rolling(window=window).corr(
                df['volume'].rolling(window=window).mean()
            )
        
        # 5. Volume momentum
        df['volume_momentum'] = df['volume'] - df['volume'].shift(7)
        
        # 6. Force Index
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_smooth'] = df['force_index'].ewm(span=13).mean()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based cyclical features"""
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Cyclical encoding (important for neural networks)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Business hours (UTC)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        return df
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        news_df: pd.DataFrame = None,
        social_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Add sentiment-based features"""
        
        # Aggregate sentiment by hour
        sentiment_hourly = pd.DataFrame(index=df.index)
        
        # News sentiment
        if news_df is not None and not news_df.empty and 'sentiment_score' in news_df.columns:
            news_df = news_df.set_index('timestamp')
            sentiment_hourly['news_sentiment'] = news_df['sentiment_score'].resample('1H').mean()
            sentiment_hourly['news_volume'] = news_df['sentiment_score'].resample('1H').count()
            sentiment_hourly['news_sentiment_std'] = news_df['sentiment_score'].resample('1H').std()
        else:
            sentiment_hourly['news_sentiment'] = 0
            sentiment_hourly['news_volume'] = 0
            sentiment_hourly['news_sentiment_std'] = 0
        
        # Social sentiment
        if social_df is not None and not social_df.empty and 'sentiment_score' in social_df.columns:
            social_df = social_df.set_index('timestamp')
            sentiment_hourly['social_sentiment'] = social_df['sentiment_score'].resample('1H').mean()
            sentiment_hourly['social_volume'] = social_df['mention_count'].resample('1H').sum()
            sentiment_hourly['social_engagement'] = social_df['engagement_score'].resample('1H').mean()
            
            # Social volume momentum
            sentiment_hourly['social_volume_change'] = social_df['mention_count'].resample('1H').sum().pct_change()
        else:
            sentiment_hourly['social_sentiment'] = 0
            sentiment_hourly['social_volume'] = 0
            sentiment_hourly['social_engagement'] = 0
            sentiment_hourly['social_volume_change'] = 0
        
        # Fill missing values
        sentiment_hourly = sentiment_hourly.fillna(method='ffill').fillna(0)
        
        # Merge with main dataframe
        df = df.join(sentiment_hourly, how='left')
        
        # Combined sentiment
        df['combined_sentiment'] = (
            df['news_sentiment'] * 0.6 + df['social_sentiment'] * 0.4
        )
        
        # Sentiment momentum (change)
        df['sentiment_momentum'] = df['combined_sentiment'].diff(periods=6)
        
        # Rolling sentiment statistics
        for window in [6, 24, 168]:
            df[f'sentiment_ma_{window}h'] = df['combined_sentiment'].rolling(window=window).mean()
            df[f'sentiment_std_{window}h'] = df['combined_sentiment'].rolling(window=window).std()
        
        return df
    
    def add_onchain_features(self, df: pd.DataFrame, onchain_df: pd.DataFrame) -> pd.DataFrame:
        """Add on-chain metrics"""
        
        if onchain_df is None or onchain_df.empty:
            return df
        
        # Resample to hourly and merge
        onchain_df = onchain_df.set_index('timestamp')
        onchain_hourly = onchain_df.resample('1H').mean().fillna(method='ffill')
        
        # Select key metrics
        onchain_features = [
            'active_addresses', 'transaction_volume', 'exchange_inflow',
            'exchange_outflow', 'mvrv_ratio', 'sopr', 'nvt_ratio'
        ]
        
        for feature in onchain_features:
            if feature in onchain_hourly.columns:
                df[feature] = onchain_hourly[feature]
                
                # Add momentum
                df[f'{feature}_momentum'] = df[feature].pct_change(periods=24)
                
                # Add rolling average
                df[f'{feature}_ma_7d'] = df[feature].rolling(window=168).mean()
        
        # Net exchange flow (outflow - inflow)
        if 'exchange_outflow' in df.columns and 'exchange_inflow' in df.columns:
            df['net_exchange_flow'] = df['exchange_outflow'] - df['exchange_inflow']
            df['exchange_flow_ratio'] = df['exchange_outflow'] / (df['exchange_inflow'] + 1)
        
        return df
    
    def add_market_features(self, df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide indicators"""
        
        if market_df is None or market_df.empty:
            return df
        
        # Broadcast market features to all timestamps
        for col in market_df.columns:
            if col != 'timestamp':
                df[col] = market_df[col].iloc[0]  # Use latest value
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Add lagged features"""
        
        # Key features to lag
        features_to_lag = ['close', 'volume', 'rsi_14', 'macd', 'combined_sentiment']
        
        for feature in features_to_lag:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features"""
        
        windows = [7, 14, 30, 168]  # 7h, 14h, 30h, 1 week
        
        for window in windows:
            # Rolling mean
            df[f'close_mean_{window}h'] = df['close'].rolling(window=window).mean()
            
            # Rolling std
            df[f'close_std_{window}h'] = df['close'].rolling(window=window).std()
            
            # Rolling min/max
            df[f'close_min_{window}h'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}h'] = df['close'].rolling(window=window).max()
            
            # Distance from min/max
            df[f'distance_from_min_{window}h'] = (df['close'] - df[f'close_min_{window}h']) / df['close']
            df[f'distance_from_max_{window}h'] = (df[f'close_max_{window}h'] - df['close']) / df['close']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def add_volatility_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-adjusted technical features"""
        
        # ATR-normalized features
        if 'atr_14' in df.columns:
            # ATR-normalized returns
            for period in [1, 6, 24]:
                if f'return_{period}h' in df.columns:
                    df[f'return_{period}h_atr_adj'] = df[f'return_{period}h'] / (df['atr_normalized'] + 1e-8)
            
            # ATR-normalized distance from moving averages
            for period in [7, 21, 50]:
                if f'distance_sma_{period}' in df.columns:
                    df[f'distance_sma_{period}_atr_adj'] = df[f'distance_sma_{period}'] / (df['atr_normalized'] + 1e-8)
        
        # Bollinger %B with volatility context
        if 'bb_percent_b_20' in df.columns and 'volatility_14d' in df.columns:
            # %B adjusted for current volatility
            volatility_zscore = (df['volatility_14d'] - df['volatility_14d'].rolling(30).mean()) / (df['volatility_14d'].rolling(30).std() + 1e-8)
            df['bb_percent_b_vol_adj'] = df['bb_percent_b_20'] * (1 + volatility_zscore * 0.1)
        
        # Risk-adjusted momentum
        for period in [5, 10, 20]:
            if f'momentum_{period}' in df.columns:
                # Sharpe-style momentum
                returns = df['close'].pct_change(period)
                volatility = df['close'].pct_change().rolling(period).std()
                df[f'momentum_{period}_sharpe'] = returns / (volatility + 1e-8)
        
        # Volatility regime classification
        if 'volatility_14d' in df.columns:
            vol_percentile = df['volatility_14d'].rolling(window=168).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
            )
            df['volatility_regime'] = pd.cut(vol_percentile, bins=[0, 0.25, 0.75, 1.0], labels=[0, 1, 2])
            df['volatility_regime'] = df['volatility_regime'].astype(float)
        
        return df
    
    def add_sentiment_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add dynamic sentiment velocity and acceleration"""
        
        if 'combined_sentiment' in df.columns:
            # Sentiment velocity (rate of change)
            for window in [1, 3, 6, 12]:
                df[f'sentiment_velocity_{window}h'] = df['combined_sentiment'].diff(window)
            
            # Sentiment acceleration
            df['sentiment_acceleration'] = df['sentiment_velocity_3h'].diff(3)
            
            # Sentiment momentum strength
            df['sentiment_momentum_strength'] = (
                df['sentiment_velocity_6h'].abs() * df['combined_sentiment'].abs()
            )
        
        if 'social_volume' in df.columns:
            # Social volume velocity
            df['social_volume_velocity'] = df['social_volume'].pct_change(6)
            
            # Volume-weighted sentiment
            if 'combined_sentiment' in df.columns:
                df['sentiment_volume_weighted'] = (
                    df['combined_sentiment'] * np.log1p(df['social_volume'])
                )
            
            # Social activity explosion detection (z-score)
            social_mean = df['social_volume'].rolling(168).mean()
            social_std = df['social_volume'].rolling(168).std()
            df['social_activity_zscore'] = (df['social_volume'] - social_mean) / (social_std + 1e-8)
        
        if 'news_volume' in df.columns:
            # News density (news per hour normalized)
            df['news_density'] = df['news_volume'] / (df['news_volume'].rolling(168).mean() + 1)
            
            # News sentiment divergence from price
            if 'combined_sentiment' in df.columns and 'return_24h' in df.columns:
                df['sentiment_price_divergence'] = df['combined_sentiment'] - np.sign(df['return_24h'])
        
        return df
    
    def add_normalized_onchain_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized on-chain metrics relative to blockchain activity"""
        
        onchain_features = ['active_addresses', 'transaction_volume', 'exchange_inflow', 
                           'exchange_outflow', 'supply_on_exchanges']
        
        for feature in onchain_features:
            if feature in df.columns:
                # Growth rate (% change)
                for window in [24, 168]:  # 1 day, 1 week
                    df[f'{feature}_growth_{window}h'] = df[feature].pct_change(window)
                
                # Z-score normalization (relative to historical activity)
                rolling_mean = df[feature].rolling(window=720).mean()  # 30 days
                rolling_std = df[feature].rolling(window=720).std()
                df[f'{feature}_zscore'] = (df[feature] - rolling_mean) / (rolling_std + 1e-8)
                
                # Percentile rank
                df[f'{feature}_percentile'] = df[feature].rolling(window=720).apply(
                    lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100 if len(x) > 1 else 0.5
                )
        
        # Network growth indicators
        if 'active_addresses' in df.columns and 'transaction_volume' in df.columns:
            # Activity per address
            df['tx_per_address'] = df['transaction_volume'] / (df['active_addresses'] + 1)
            df['tx_per_address_growth'] = df['tx_per_address'].pct_change(168)
        
        # Exchange flow indicators
        if 'exchange_inflow' in df.columns and 'exchange_outflow' in df.columns:
            # Net flow normalized by total exchange balance
            df['exchange_net_flow_ratio'] = (
                (df['exchange_outflow'] - df['exchange_inflow']) / 
                (df['exchange_inflow'] + df['exchange_outflow'] + 1e-8)
            )
            
            # Flow momentum
            df['exchange_flow_momentum'] = df['exchange_net_flow_ratio'].diff(24)
        
        # MVRV momentum
        if 'mvrv_ratio' in df.columns:
            df['mvrv_momentum'] = df['mvrv_ratio'].diff(168)
            df['mvrv_velocity'] = df['mvrv_ratio'].pct_change(24)
        
        # SOPR trend
        if 'sopr' in df.columns:
            # SOPR deviation from 1 (profit/loss threshold)
            df['sopr_deviation'] = df['sopr'] - 1.0
            df['sopr_trend'] = df['sopr'].rolling(7).mean() - df['sopr'].rolling(30).mean()
        
        return df
    
    def add_multi_horizon_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features across multiple time horizons"""
        
        # Multi-horizon returns (1h, 3h, 6h, 12h, 24h, 48h, 168h)
        horizons = [1, 3, 6, 12, 24, 48, 168]
        
        for h in horizons:
            if h == 1 or f'return_{h}h' not in df.columns:
                df[f'return_{h}h'] = df['close'].pct_change(h)
            
            # Log returns for better distribution
            df[f'log_return_{h}h'] = np.log1p(df[f'return_{h}h'])
        
        # Momentum consistency score (how many horizons agree on direction)
        momentum_cols = [f'return_{h}h' for h in horizons if f'return_{h}h' in df.columns]
        if momentum_cols:
            df['momentum_consistency'] = df[momentum_cols].apply(
                lambda row: (row > 0).sum() / len(row), axis=1
            )
            
            # Momentum strength (average magnitude across horizons)
            df['momentum_strength'] = df[momentum_cols].abs().mean(axis=1)
        
        # Acceleration across horizons
        df['momentum_acceleration_short'] = df['return_1h'] - df['return_6h'] / 6
        df['momentum_acceleration_long'] = df['return_24h'] - df['return_168h'] / 7
        
        # Trend alignment (short-term vs long-term)
        if 'return_24h' in df.columns and 'return_168h' in df.columns:
            df['trend_alignment'] = np.sign(df['return_24h']) * np.sign(df['return_168h'])
        
        # Price distance from multiple moving averages
        ma_periods = [7, 21, 50, 100, 200]
        for period in ma_periods:
            if f'sma_{period}' in df.columns:
                df[f'price_to_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Golden/Death cross signals
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']
            df['golden_cross_strength'] = (df['sma_50_200_ratio'] - 1) * 100
        
        # Volume-adjusted momentum
        if 'volume' in df.columns:
            for h in [1, 6, 24]:
                if f'return_{h}h' in df.columns:
                    volume_ratio = df['volume'] / df['volume'].rolling(h).mean()
                    df[f'volume_adjusted_return_{h}h'] = df[f'return_{h}h'] * np.log1p(volume_ratio)
        
        return df


if __name__ == "__main__":
    """Test feature engineering"""
    
    # Create sample data
    from data_collector import CryptoDataCollector
    
    collector = CryptoDataCollector()
    data = collector.collect_all_data('bitcoin', hours_back=720)
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_all_features(
        price_df=data['price'],
        news_df=data['news'],
        social_df=data['social'],
        onchain_df=data['onchain'],
        market_df=data['market']
    )
    
    print("\n" + "="*60)
    print("Feature Engineering Summary")
    print("="*60)
    print(f"Total features created: {len(features_df.columns)}")
    print(f"Total samples: {len(features_df)}")
    print(f"\nFeature categories:")
    
    categories = {
        'Technical Indicators': [c for c in features_df.columns if any(x in c for x in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'stoch'])],
        'Price Patterns': [c for c in features_df.columns if any(x in c for x in ['return', 'trend', 'volatility', 'wick', 'range'])],
        'Volume Features': [c for c in features_df.columns if 'volume' in c],
        'Sentiment Features': [c for c in features_df.columns if 'sentiment' in c or 'news' in c or 'social' in c],
        'On-chain Metrics': [c for c in features_df.columns if any(x in c for x in ['address', 'transaction', 'exchange', 'mvrv', 'sopr', 'nvt'])],
        'Time Features': [c for c in features_df.columns if any(x in c for x in ['hour', 'day', 'month', 'weekend'])],
    }
    
    for category, features in categories.items():
        print(f"  {category}: {len(features)} features")
    
    print(f"\nSample features:")
    print(features_df.head())

    # Persist simplified features for the dashboard (best-effort)
    try:
        os.makedirs('cache', exist_ok=True)
        price_df = data.get('price')
        if price_df is not None and isinstance(price_df, pd.DataFrame) and len(price_df) > 0:
            simple_df = add_simple_features(price_df, news_df=data.get('news'))
            simple_path = os.path.join('cache', 'simple_features.pkl')
            simple_df.to_pickle(simple_path)

            meta_path = os.path.join('cache', 'artifacts_meta.json')
            meta = {
                'saved_at_utc': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'artifact': 'simple_features.pkl',
                'rows': int(len(simple_df)),
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                import json

                json.dump(meta, f, indent=2)

            print(f"\nSaved dashboard features: {simple_path}")
    except Exception as e:
        print(f"\nWarning: failed to save simple_features.pkl: {e}")
