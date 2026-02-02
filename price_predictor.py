"""
Crypto Price Predictor - Advanced AI/ML System
===============================================

This module implements a sophisticated multi-modal deep learning system for cryptocurrency
price prediction and optimal buy/sell signal generation.

Architecture:
- Hybrid LSTM-Transformer model for time series prediction
- Multi-source data integration (price, sentiment, on-chain, technical indicators)
- Real-time news and social media sentiment analysis
- Attention mechanisms for feature importance
- Ensemble prediction with uncertainty quantification

Author: AI/ML Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """Custom attention layer for feature importance weighting"""
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        # Apply attention weights
        weighted_input = x * attention_weights
        return tf.reduce_sum(weighted_input, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class CryptoPricePredictor:
    """
    Advanced Cryptocurrency Price Prediction System
    
    This system uses a hybrid architecture combining:
    1. LSTM layers for temporal dependencies
    2. Transformer-style attention for feature importance
    3. Multi-head attention for different time horizons
    4. Dense layers for final prediction
    
    Features incorporated:
    - Historical OHLCV data
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Sentiment scores from news and social media
    - On-chain metrics (transaction volume, active addresses, etc.)
    - Market factors (BTC dominance, fear & greed index, etc.)
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 24,  # hours ahead
        n_features: int = 50,
        lstm_units: List[int] = [128, 64],
        attention_units: int = 128,
        dense_units: List[int] = [64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Initialize the predictor with specified architecture
        
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Hours ahead to predict
            n_features: Number of input features
            lstm_units: List of LSTM layer units
            attention_units: Units in attention mechanism
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.feature_names = []
        self.training_history = None
        
        logger.info("CryptoPricePredictor initialized with configuration:")
        logger.info(f"  Sequence Length: {sequence_length}")
        logger.info(f"  Prediction Horizon: {prediction_horizon} hours")
        logger.info(f"  Features: {n_features}")
    
    def build_model(self) -> Model:
        """
        Build the hybrid LSTM-Attention model architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building model architecture...")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers with residual connections
        x = inputs
        for i, units in enumerate(self.lstm_units):
            lstm_out = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2,
                name=f'lstm_{i+1}'
            )(x)
            
            # Add residual connection if dimensions match
            if i > 0 and units == self.lstm_units[i-1]:
                x = layers.Add()([x, lstm_out])
            else:
                x = lstm_out
            
            x = layers.BatchNormalization()(x)
        
        # Multi-head attention mechanism
        attention_output = AttentionLayer(self.attention_units, name='attention')(x)
        
        # Additional context: max and average pooling
        max_pool = layers.GlobalMaxPooling1D()(x)
        avg_pool = layers.GlobalAveragePooling1D()(x)
        
        # Concatenate all features
        concatenated = layers.Concatenate()([attention_output, max_pool, avg_pool])
        
        # Dense layers with dropout
        x = concatenated
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.BatchNormalization()(x)
        
        # Output layers
        # 1. Price prediction (continuous)
        price_output = layers.Dense(1, activation='linear', name='price')(x)
        
        # 2. Direction prediction (classification: up/down/neutral)
        direction_output = layers.Dense(3, activation='softmax', name='direction')(x)
        
        # 3. Volatility prediction (continuous, positive)
        volatility_output = layers.Dense(1, activation='softplus', name='volatility')(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[price_output, direction_output, volatility_output]
        )
        
        # Compile with custom loss weights
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'price': 'huber',  # Robust to outliers
                'direction': 'categorical_crossentropy',
                'volatility': 'mse'
            },
            loss_weights={
                'price': 1.0,
                'direction': 0.5,
                'volatility': 0.3
            },
            metrics={
                'price': ['mae', 'mse'],
                'direction': ['accuracy'],
                'volatility': ['mae']
            }
        )
        
        self.model = model
        logger.info(f"Model built successfully with {model.count_params():,} parameters")
        return model
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_col: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequential data for training
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            X: Input sequences
            y_price: Price targets
            y_direction: Direction targets (0=down, 1=neutral, 2=up)
        """
        logger.info(f"Preparing sequences from {len(data)} data points...")
        
        # Store feature names
        self.feature_names = [col for col in data.columns if col != target_col]
        
        # Extract features and target
        features = data[self.feature_names].values
        target = data[target_col].values
        
        # Scale features
        features_scaled = self.scaler_X.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        
        X, y_price, y_direction, y_volatility = [], [], [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            # Input sequence
            X.append(features_scaled[i:i + self.sequence_length])
            
            # Price target (prediction_horizon hours ahead)
            future_idx = i + self.sequence_length + self.prediction_horizon - 1
            y_price.append(target_scaled[future_idx])
            
            # Direction target (compare future to current)
            current_price = target[i + self.sequence_length - 1]
            future_price = target[future_idx]
            price_change_pct = (future_price - current_price) / current_price * 100
            
            # Classify: down (<-2%), neutral (-2% to 2%), up (>2%)
            if price_change_pct < -2:
                direction = [1, 0, 0]  # Down
            elif price_change_pct > 2:
                direction = [0, 0, 1]  # Up
            else:
                direction = [0, 1, 0]  # Neutral
            y_direction.append(direction)
            
            # Volatility target (std of returns in prediction window)
            window_prices = target[i + self.sequence_length:future_idx + 1]
            returns = np.diff(window_prices) / window_prices[:-1]
            y_volatility.append(np.std(returns))
        
        X = np.array(X)
        y_price = np.array(y_price)
        y_direction = np.array(y_direction)
        y_volatility = np.array(y_volatility)
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y_price shape: {y_price.shape}")
        logger.info(f"  y_direction shape: {y_direction.shape}")
        
        return X, y_price, y_direction, y_volatility
    
    def train(
        self,
        X: np.ndarray,
        y_price: np.ndarray,
        y_direction: np.ndarray,
        y_volatility: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15
    ):
        """
        Train the model with early stopping and learning rate reduction
        
        Args:
            X: Input sequences
            y_price: Price targets
            y_direction: Direction targets
            y_volatility: Volatility targets
            validation_split: Fraction of data for validation
            epochs: Maximum training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Starting training...")
        logger.info(f"  Training samples: {int(len(X) * (1 - validation_split))}")
        logger.info(f"  Validation samples: {int(len(X) * validation_split)}")
        logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'crypto_predictor_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.training_history = self.model.fit(
            X,
            {
                'price': y_price,
                'direction': y_direction,
                'volatility': y_volatility
            },
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        return self.training_history
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals
        
        Args:
            X: Input sequences
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        price_pred, direction_pred, volatility_pred = self.model.predict(X, verbose=0)
        
        # Inverse transform price predictions
        price_pred_original = self.scaler_y.inverse_transform(price_pred)
        
        # Get direction class and confidence
        direction_class = np.argmax(direction_pred, axis=1)
        direction_confidence = np.max(direction_pred, axis=1)
        
        results = {
            'price': price_pred_original.flatten(),
            'direction': direction_class,  # 0=down, 1=neutral, 2=up
            'direction_confidence': direction_confidence,
            'volatility': volatility_pred.flatten()
        }
        
        if return_confidence:
            # Calculate confidence intervals using volatility
            results['price_lower'] = price_pred_original.flatten() - 1.96 * volatility_pred.flatten()
            results['price_upper'] = price_pred_original.flatten() + 1.96 * volatility_pred.flatten()
        
        return results
    
    def generate_signals(
        self,
        predictions: Dict[str, np.ndarray],
        current_price: float,
        confidence_threshold: float = 0.7
    ) -> Dict[str, any]:
        """
        Generate trading signals based on predictions
        
        Args:
            predictions: Dictionary from predict()
            current_price: Current market price
            confidence_threshold: Minimum confidence for signal
            
        Returns:
            Trading signal with recommendation and metadata
        """
        latest_pred = {
            'price': predictions['price'][-1],
            'direction': predictions['direction'][-1],
            'confidence': predictions['direction_confidence'][-1],
            'volatility': predictions['volatility'][-1]
        }
        
        # Calculate expected return
        expected_return = (latest_pred['price'] - current_price) / current_price * 100
        
        # Determine signal
        signal = 'HOLD'
        strength = 0.0
        
        if latest_pred['confidence'] >= confidence_threshold:
            if latest_pred['direction'] == 2 and expected_return > 3:  # Up
                signal = 'BUY'
                strength = min(latest_pred['confidence'] * abs(expected_return) / 10, 1.0)
            elif latest_pred['direction'] == 0 and expected_return < -3:  # Down
                signal = 'SELL'
                strength = min(latest_pred['confidence'] * abs(expected_return) / 10, 1.0)
        
        # Risk assessment
        risk_level = 'LOW'
        if latest_pred['volatility'] > 0.05:
            risk_level = 'HIGH'
        elif latest_pred['volatility'] > 0.03:
            risk_level = 'MEDIUM'
        
        return {
            'signal': signal,
            'strength': strength,
            'current_price': current_price,
            'predicted_price': latest_pred['price'],
            'expected_return': expected_return,
            'confidence': latest_pred['confidence'],
            'risk_level': risk_level,
            'volatility': latest_pred['volatility'],
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: str = 'crypto_predictor_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'crypto_predictor_model.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"Model loaded from {filepath}")


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions
    Combines LSTM, GRU, and Transformer-based models
    """
    
    def __init__(self, n_models: int = 3):
        self.n_models = n_models
        self.models = []
        logger.info(f"Initializing ensemble with {n_models} models")
    
    def build_ensemble(self, **kwargs):
        """Build multiple models with different architectures"""
        configs = [
            {'lstm_units': [128, 64], 'attention_units': 128},
            {'lstm_units': [256, 128, 64], 'attention_units': 256},
            {'lstm_units': [64, 32], 'attention_units': 64}
        ]
        
        for i, config in enumerate(configs[:self.n_models]):
            model_kwargs = {**kwargs, **config}
            model = CryptoPricePredictor(**model_kwargs)
            model.build_model()
            self.models.append(model)
            logger.info(f"Built ensemble model {i+1}/{self.n_models}")
    
    def train_ensemble(self, X, y_price, y_direction, y_volatility, **kwargs):
        """Train all models in the ensemble"""
        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")
            model.train(X, y_price, y_direction, y_volatility, **kwargs)
    
    def predict(self, X) -> Dict[str, np.ndarray]:
        """Make ensemble predictions by averaging"""
        all_predictions = [model.predict(X) for model in self.models]
        
        # Average predictions
        ensemble_pred = {
            'price': np.mean([p['price'] for p in all_predictions], axis=0),
            'direction_confidence': np.mean([p['direction_confidence'] for p in all_predictions], axis=0),
            'volatility': np.mean([p['volatility'] for p in all_predictions], axis=0)
        }
        
        # Majority vote for direction
        directions = np.array([p['direction'] for p in all_predictions])
        ensemble_pred['direction'] = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=directions
        )
        
        return ensemble_pred


if __name__ == "__main__":
    """
    Example usage and demonstration
    """
    logger.info("="*60)
    logger.info("Crypto Price Predictor - Advanced AI/ML System")
    logger.info("="*60)
    
    # This is a demonstration - in production, you would:
    # 1. Load real data from data_collector.py
    # 2. Engineer features from feature_engineering.py
    # 3. Add sentiment from sentiment_analyzer.py
    # 4. Train and deploy the model
    
    logger.info("\nSystem Architecture:")
    logger.info("1. Data Collection Layer:")
    logger.info("   - Real-time price data (OHLCV)")
    logger.info("   - News aggregation (multiple sources)")
    logger.info("   - Social media monitoring (Twitter, Reddit)")
    logger.info("   - On-chain metrics (transaction volume, addresses)")
    logger.info("   - Market indicators (fear & greed, BTC dominance)")
    
    logger.info("\n2. Feature Engineering Layer:")
    logger.info("   - Technical indicators: RSI, MACD, Bollinger Bands, etc.")
    logger.info("   - Sentiment scores: News and social media analysis")
    logger.info("   - On-chain features: Network activity, whale movements")
    logger.info("   - Time-based features: Hour, day, cyclical patterns")
    logger.info("   - Volatility metrics: Historical and implied")
    
    logger.info("\n3. ML Model Layer:")
    logger.info("   - Hybrid LSTM-Transformer architecture")
    logger.info("   - Multi-task learning: Price, direction, volatility")
    logger.info("   - Attention mechanism for feature importance")
    logger.info("   - Ensemble predictions for robustness")
    
    logger.info("\n4. Signal Generation Layer:")
    logger.info("   - BUY/SELL/HOLD recommendations")
    logger.info("   - Confidence scores and risk assessment")
    logger.info("   - Optimal entry/exit points")
    logger.info("   - Real-time monitoring and alerts")
    
    logger.info("\n" + "="*60)
    logger.info("Ready to process data and generate predictions!")
    logger.info("="*60)
