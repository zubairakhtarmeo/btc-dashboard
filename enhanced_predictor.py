"""
Enhanced Price Predictor with Advanced Architecture
===================================================

Improvements:
- GRU layers alongside LSTM
- Multi-head attention
- Temporal Convolutional Networks (TCN)
- Monte Carlo dropout for uncertainty
- Mixed precision training
- Dynamic thresholds for classification
- Multi-horizon prediction
- Heterogeneous ensemble
"""

import os
# CRITICAL: Set TensorFlow environment BEFORE importing TensorFlow
# This prevents CUDA initialization crashes on Streamlit Cloud
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle

logger = logging.getLogger(__name__)


class TemporalConvLayer(layers.Layer):
    """Temporal Convolutional Network layer"""
    
    def __init__(self, filters=64, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
    def build(self, input_shape):
        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.norm = layers.BatchNormalization()
        
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class MultiHeadAttentionCustom(layers.Layer):
    """Multi-head attention for different time horizons"""
    
    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=0.1
        )
        self.layernorm = layers.LayerNormalization()
        
    def call(self, x, training=False):
        attn_output = self.attention(x, x, training=training)
        return self.layernorm(x + attn_output)


class EnhancedCryptoPricePredictor:
    """
    Enhanced cryptocurrency price predictor with advanced architecture
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = [1, 6, 24],  # Multi-horizon
        n_features: int = 50,
        architecture: str = 'hybrid',  # 'lstm', 'gru', 'tcn', 'hybrid'
        lstm_units: List[int] = [128, 64],
        gru_units: List[int] = [128, 64],
        tcn_filters: List[int] = [64, 64],
        attention_heads: int = 4,
        attention_key_dim: int = 32,
        dense_units: List[int] = [64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        use_mixed_precision: bool = True,
        mc_dropout_samples: int = 10,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        crash_threshold_pct: float = 5.0,
        pump_threshold_pct: float = 5.0,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.n_features = n_features
        self.architecture = architecture
        self.lstm_units = lstm_units
        self.gru_units = gru_units
        self.tcn_filters = tcn_filters
        self.attention_heads = attention_heads
        self.attention_key_dim = attention_key_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.mc_dropout_samples = mc_dropout_samples
        self.quantiles = list(quantiles or [0.1, 0.5, 0.9])
        self.crash_threshold_pct = float(crash_threshold_pct)
        self.pump_threshold_pct = float(pump_threshold_pct)
        
        # Enable mixed precision for faster training
        if use_mixed_precision and tf.config.list_physical_devices('GPU'):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
        
        self.model = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.feature_names = []
        self.training_history = None
        
        # Dynamic thresholds for direction classification
        self.dynamic_thresholds = {'up': 2.0, 'down': -2.0}  # Will be updated
        
        logger.info(f"EnhancedCryptoPricePredictor initialized ({architecture} architecture)")

    @staticmethod
    def _pinball_loss(q: float):
        """Quantile (pinball) loss for a fixed quantile q in (0,1)."""
        q = float(q)

        def loss(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

        return loss
    
    def build_model(self) -> Model:
        """Build enhanced hybrid model"""
        logger.info(f"Building {self.architecture} model...")
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        if self.architecture == 'hybrid':
            # Parallel pathways for different temporal patterns
            lstm_path = self._build_lstm_path(inputs)
            gru_path = self._build_gru_path(inputs)
            tcn_path = self._build_tcn_path(inputs)
            
            # Concatenate parallel paths
            x = layers.Concatenate()([lstm_path, gru_path, tcn_path])
            
        elif self.architecture == 'lstm':
            x = self._build_lstm_path(inputs)
        elif self.architecture == 'gru':
            x = self._build_gru_path(inputs)
        elif self.architecture == 'tcn':
            x = self._build_tcn_path(inputs)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Multi-head attention
        attention_output = MultiHeadAttentionCustom(
            num_heads=self.attention_heads,
            key_dim=self.attention_key_dim
        )(inputs)
        
        # Global pooling on attention output
        attention_pooled = layers.GlobalAveragePooling1D()(attention_output)
        
        # Combine with pathway output
        x = layers.Concatenate()([x, attention_pooled])
        
        # Dense layers with MC Dropout
        for units in self.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # Always on for MC
            x = layers.BatchNormalization()(x)
        
        # Multi-horizon outputs (dict outputs so we can train/predict by name)
        outputs: Dict[str, layers.Layer] = {}
        for horizon in self.prediction_horizons:
            # Point estimate (scaled price)
            outputs[f'price_{horizon}h'] = layers.Dense(1, activation='linear', name=f'price_{horizon}h')(x)

            # Quantiles (scaled price)
            for q in self.quantiles:
                q_label = int(round(float(q) * 100))
                outputs[f'price_p{q_label}_{horizon}h'] = layers.Dense(
                    1,
                    activation='linear',
                    name=f'price_p{q_label}_{horizon}h',
                )(x)

            # Direction (up/neutral/down)
            outputs[f'direction_{horizon}h'] = layers.Dense(3, activation='softmax', name=f'direction_{horizon}h')(x)

            # Volatility
            outputs[f'volatility_{horizon}h'] = layers.Dense(1, activation='softplus', name=f'volatility_{horizon}h')(x)

            # Tail-event risk heads
            outputs[f'crash_{horizon}h'] = layers.Dense(1, activation='sigmoid', name=f'crash_{horizon}h')(x)
            outputs[f'pump_{horizon}h'] = layers.Dense(1, activation='sigmoid', name=f'pump_{horizon}h')(x)

        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with loss weights
        loss_dict: Dict[str, object] = {}
        loss_weights_dict: Dict[str, float] = {}
        metrics_dict: Dict[str, list] = {}

        for horizon in self.prediction_horizons:
            # Shorter horizons get slightly more weight
            weight_factor = 1.0 / float(horizon)

            loss_dict[f'price_{horizon}h'] = 'huber'
            loss_weights_dict[f'price_{horizon}h'] = weight_factor
            metrics_dict[f'price_{horizon}h'] = ['mae']

            for q in self.quantiles:
                q_label = int(round(float(q) * 100))
                key = f'price_p{q_label}_{horizon}h'
                loss_dict[key] = self._pinball_loss(q)
                loss_weights_dict[key] = weight_factor * 0.6

            loss_dict[f'direction_{horizon}h'] = 'categorical_crossentropy'
            loss_weights_dict[f'direction_{horizon}h'] = weight_factor * 0.5
            metrics_dict[f'direction_{horizon}h'] = ['accuracy']

            loss_dict[f'volatility_{horizon}h'] = 'mse'
            loss_weights_dict[f'volatility_{horizon}h'] = weight_factor * 0.3
            metrics_dict[f'volatility_{horizon}h'] = ['mae']

            loss_dict[f'crash_{horizon}h'] = 'binary_crossentropy'
            loss_weights_dict[f'crash_{horizon}h'] = weight_factor * 0.35
            metrics_dict[f'crash_{horizon}h'] = ['accuracy']

            loss_dict[f'pump_{horizon}h'] = 'binary_crossentropy'
            loss_weights_dict[f'pump_{horizon}h'] = weight_factor * 0.35
            metrics_dict[f'pump_{horizon}h'] = ['accuracy']
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict
        )
        
        self.model = model
        logger.info(f"Model built with {model.count_params():,} parameters")
        return model
    
    def _build_lstm_path(self, inputs):
        """Build LSTM pathway"""
        x = inputs
        for i, units in enumerate(self.lstm_units):
            x = layers.LSTM(
                units,
                return_sequences=(i < len(self.lstm_units) - 1),
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2
            )(x)
            x = layers.BatchNormalization()(x)
        return x
    
    def _build_gru_path(self, inputs):
        """Build GRU pathway (often faster and more efficient)"""
        x = inputs
        for i, units in enumerate(self.gru_units):
            x = layers.GRU(
                units,
                return_sequences=(i < len(self.gru_units) - 1),
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2
            )(x)
            x = layers.BatchNormalization()(x)
        return x
    
    def _build_tcn_path(self, inputs):
        """Build TCN pathway (lightweight alternative)"""
        x = inputs
        for filters in self.tcn_filters:
            x = TemporalConvLayer(filters=filters, kernel_size=3, dilation_rate=2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        return x
    
    def prepare_sequences_multihorizon(
        self,
        data: pd.DataFrame,
        target_col: str = 'close'
    ) -> Tuple:
        """Prepare sequences for multi-horizon prediction"""
        logger.info(f"Preparing multi-horizon sequences...")
        
        self.feature_names = [col for col in data.columns if col != target_col]
        
        features = data[self.feature_names].values
        target = data[target_col].values
        
        # Scale
        features_scaled = self.scaler_X.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Calculate dynamic thresholds based on historical volatility
        returns = np.diff(target) / target[:-1] * 100
        volatility = np.std(returns)
        self.dynamic_thresholds = {
            'up': volatility * 0.5,  # 0.5 standard deviations
            'down': -volatility * 0.5
        }
        logger.info(f"Dynamic thresholds: up={self.dynamic_thresholds['up']:.2f}%, down={self.dynamic_thresholds['down']:.2f}%")
        
        X, y_dict = [], {}
        max_horizon = max(self.prediction_horizons)
        
        # Initialize output dictionaries
        for horizon in self.prediction_horizons:
            y_dict[f'price_{horizon}h'] = []
            for q in self.quantiles:
                q_label = int(round(float(q) * 100))
                y_dict[f'price_p{q_label}_{horizon}h'] = []
            y_dict[f'direction_{horizon}h'] = []
            y_dict[f'volatility_{horizon}h'] = []
            y_dict[f'crash_{horizon}h'] = []
            y_dict[f'pump_{horizon}h'] = []
        
        for i in range(len(data) - self.sequence_length - max_horizon):
            X.append(features_scaled[i:i + self.sequence_length])
            
            current_price = target[i + self.sequence_length - 1]
            
            for horizon in self.prediction_horizons:
                future_idx = i + self.sequence_length + horizon - 1
                future_price = target[future_idx]
                
                # Price target
                y_val = float(target_scaled[future_idx])
                y_dict[f'price_{horizon}h'].append(y_val)
                for q in self.quantiles:
                    q_label = int(round(float(q) * 100))
                    y_dict[f'price_p{q_label}_{horizon}h'].append(y_val)
                
                # Direction target (using dynamic thresholds)
                pct_change = ((future_price - current_price) / current_price) * 100
                if pct_change > self.dynamic_thresholds['up']:
                    direction = [0, 0, 1]  # Up
                elif pct_change < self.dynamic_thresholds['down']:
                    direction = [1, 0, 0]  # Down
                else:
                    direction = [0, 1, 0]  # Neutral
                y_dict[f'direction_{horizon}h'].append(direction)

                # Crash/pump targets (tail event classification)
                y_dict[f'crash_{horizon}h'].append(1.0 if pct_change <= -self.crash_threshold_pct else 0.0)
                y_dict[f'pump_{horizon}h'].append(1.0 if pct_change >= self.pump_threshold_pct else 0.0)
                
                # Volatility target (std of returns in horizon window)
                if future_idx + horizon < len(target):
                    window = target[future_idx:future_idx + horizon]
                    window_returns = np.diff(window) / window[:-1]
                    volatility = np.std(window_returns) if len(window_returns) > 0 else 0
                else:
                    volatility = 0
                y_dict[f'volatility_{horizon}h'].append(volatility)
        
        X = np.array(X)

        for key in list(y_dict.keys()):
            arr = np.array(y_dict[key])
            # Dense(1) heads expect shape (N, 1)
            if key.startswith('price_') and not key.startswith('price_p'):
                arr = arr.reshape(-1, 1)
            elif key.startswith('price_p'):
                arr = arr.reshape(-1, 1)
            elif key.startswith('volatility_'):
                arr = arr.reshape(-1, 1)
            elif key.startswith('crash_') or key.startswith('pump_'):
                arr = arr.reshape(-1, 1)
            y_dict[key] = arr
        
        logger.info(f"Created {len(X)} sequences with {max_horizon}h max horizon")
        return X, y_dict
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = None
    ) -> Dict[str, np.ndarray]:
        """Predict with Monte Carlo dropout for uncertainty estimation"""
        if n_samples is None:
            n_samples = self.mc_dropout_samples
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred)

        # Support both:
        # - New models: dict outputs (preferred)
        # - Legacy models: list outputs in groups of 3 per horizon
        first = predictions[0]
        results: Dict[str, np.ndarray] = {}

        if isinstance(first, dict):
            keys = list(first.keys())
            for k in keys:
                stacked = np.stack([p[k] for p in predictions], axis=0)
                results[k] = np.mean(stacked, axis=0)

            # For backwards-compat UI: add std for main price + direction
            for horizon in self.prediction_horizons:
                pk = f'price_{horizon}h'
                dk = f'direction_{horizon}h'
                if pk in first:
                    stacked = np.stack([p[pk] for p in predictions], axis=0)
                    results[f'{pk}_std'] = np.std(stacked, axis=0)
                if dk in first:
                    stacked = np.stack([p[dk] for p in predictions], axis=0)
                    results[f'{dk}_std'] = np.std(stacked, axis=0)

            return results

        # Legacy list output handling
        for i, horizon in enumerate(self.prediction_horizons):
            price_preds = [p[i * 3] for p in predictions]
            dir_preds = [p[i * 3 + 1] for p in predictions]
            vol_preds = [p[i * 3 + 2] for p in predictions]

            results[f'price_{horizon}h'] = np.mean(price_preds, axis=0)
            results[f'price_{horizon}h_std'] = np.std(price_preds, axis=0)

            results[f'direction_{horizon}h'] = np.mean(dir_preds, axis=0)
            results[f'direction_{horizon}h_std'] = np.std(dir_preds, axis=0)

            results[f'volatility_{horizon}h'] = np.mean(vol_preds, axis=0)

        return results
    
    def save_model_full(self, base_path: str = 'models/enhanced_predictor'):
        """Save model with all components"""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        # Save model
        self.model.save(f"{base_path}_model.h5")
        
        # Save scalers and metadata
        metadata = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'dynamic_thresholds': self.dynamic_thresholds,
            'config': {
                'sequence_length': self.sequence_length,
                'prediction_horizons': self.prediction_horizons,
                'n_features': self.n_features,
                'architecture': self.architecture,
                'quantiles': self.quantiles,
                'crash_threshold_pct': self.crash_threshold_pct,
                'pump_threshold_pct': self.pump_threshold_pct,
                'output_schema_version': 2,
            }
        }
        
        with open(f"{base_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Full model saved to {base_path}")


class HeterogeneousEnsemble:
    """
    Heterogeneous ensemble with performance-weighted predictions
    Combines LSTM, GRU, TCN, and Hybrid architectures
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = [1, 6, 24],
        n_features: int = 50,
        **kwargs
    ):
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.n_features = n_features
        self.models = []
        self.model_weights = []
        self.model_architectures = ['lstm', 'gru', 'tcn', 'hybrid']
        
        logger.info(f"Initializing heterogeneous ensemble with {len(self.model_architectures)} models")
    
    def build_ensemble(self, **kwargs):
        """Build diverse model architectures"""
        for arch in self.model_architectures:
            logger.info(f"Building {arch} model...")
            model = EnhancedCryptoPricePredictor(
                sequence_length=self.sequence_length,
                prediction_horizons=self.prediction_horizons,
                n_features=self.n_features,
                architecture=arch,
                **kwargs
            )
            model.build_model()
            self.models.append(model)
        
        # Initialize equal weights
        self.model_weights = [1.0 / len(self.models)] * len(self.models)
        logger.info("Ensemble built successfully")
    
    def train_ensemble_parallel(
        self,
        X: np.ndarray,
        y_dict: Dict,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        use_time_series_cv: bool = True
    ):
        """Train ensemble with time-series cross-validation"""
        logger.info("Training heterogeneous ensemble...")
        
        if use_time_series_cv:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = [[] for _ in self.models]
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"Training fold {fold + 1}/3")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = {k: v[train_idx] for k, v in y_dict.items()}
                y_val = {k: v[val_idx] for k, v in y_dict.items()}
                
                for i, model in enumerate(self.models):
                    logger.info(f"  Training {model.architecture} model...")

                    y_train_fit = y_train
                    y_val_fit = y_val
                    if isinstance(model.model.output, dict):
                        y_train_fit = {name: y_train[name] for name in model.model.output_names if name in y_train}
                        y_val_fit = {name: y_val[name] for name in model.model.output_names if name in y_val}
                    
                    # Prepare y for this model
                    y_train_list = []
                    y_val_list = []
                    for horizon in model.prediction_horizons:
                        y_train_list.extend([
                            y_train[f'price_{horizon}h'],
                            y_train[f'direction_{horizon}h'],
                            y_train[f'volatility_{horizon}h']
                        ])
                        y_val_list.extend([
                            y_val[f'price_{horizon}h'],
                            y_val[f'direction_{horizon}h'],
                            y_val[f'volatility_{horizon}h']
                        ])
                    
                    history = model.model.fit(
                        X_train,
                        (y_train_fit if isinstance(model.model.output, dict) else y_train_list),
                        validation_data=(X_val, (y_val_fit if isinstance(model.model.output, dict) else y_val_list)),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                        ],
                        verbose=0
                    )
                    
                    # Track validation performance
                    val_loss = min(history.history['val_loss'])
                    val_scores[i].append(val_loss)
            
            # Calculate performance-based weights (inverse of validation loss)
            avg_scores = [np.mean(scores) for scores in val_scores]
            inverse_scores = [1.0 / (score + 1e-8) for score in avg_scores]
            total = sum(inverse_scores)
            self.model_weights = [w / total for w in inverse_scores]
            
            logger.info("Performance-based weights:")
            for arch, weight in zip(self.model_architectures, self.model_weights):
                logger.info(f"  {arch}: {weight:.3f}")
        
        else:
            # Simple train on full data
            for i, model in enumerate(self.models):
                logger.info(f"Training {model.architecture} model...")

                if isinstance(model.model.output, dict):
                    y_train = {name: y_dict[name] for name in model.model.output_names if name in y_dict}
                    model.model.fit(
                        X,
                        y_train,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
                        ],
                        verbose=1
                    )
                    continue

                y_train_list = []
                for horizon in model.prediction_horizons:
                    y_train_list.extend([
                        y_dict[f'price_{horizon}h'],
                        y_dict[f'direction_{horizon}h'],
                        y_dict[f'volatility_{horizon}h']
                    ])

                model.model.fit(
                    X,
                    y_train_list,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
                    ],
                    verbose=1
                )
    
    def predict_ensemble(
        self,
        X: np.ndarray,
        use_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """Weighted ensemble prediction"""
        all_predictions = []
        
        for model, weight in zip(self.models, self.model_weights):
            if use_uncertainty:
                pred = model.predict_with_uncertainty(X)
            else:
                pred_raw = model.model.predict(X, verbose=0)
                if isinstance(pred_raw, dict):
                    pred = pred_raw
                else:
                    pred = {}
                    for i, horizon in enumerate(model.prediction_horizons):
                        pred[f'price_{horizon}h'] = pred_raw[i*3]
                        pred[f'direction_{horizon}h'] = pred_raw[i*3+1]
                        pred[f'volatility_{horizon}h'] = pred_raw[i*3+2]
            
            all_predictions.append((pred, weight))
        
        # Weighted average
        ensemble_pred = {}
        
        for horizon in self.prediction_horizons:
            # Core heads
            price_key = f'price_{horizon}h'
            dir_key = f'direction_{horizon}h'
            vol_key = f'volatility_{horizon}h'
            crash_key = f'crash_{horizon}h'
            pump_key = f'pump_{horizon}h'

            if all(price_key in p[0] for p in all_predictions):
                price_preds = [p[0][price_key] * p[1] for p in all_predictions]
                ensemble_pred[price_key] = np.sum(price_preds, axis=0)

            if all(dir_key in p[0] for p in all_predictions):
                dir_preds = [p[0][dir_key] * p[1] for p in all_predictions]
                ensemble_pred[dir_key] = np.sum(dir_preds, axis=0)
                ensemble_pred[f'{dir_key}_class'] = np.argmax(ensemble_pred[dir_key], axis=-1)

            if all(vol_key in p[0] for p in all_predictions):
                vol_preds = [p[0][vol_key] * p[1] for p in all_predictions]
                ensemble_pred[vol_key] = np.sum(vol_preds, axis=0)

            # Tail heads
            if all(crash_key in p[0] for p in all_predictions):
                crash_preds = [p[0][crash_key] * p[1] for p in all_predictions]
                ensemble_pred[crash_key] = np.sum(crash_preds, axis=0)
            if all(pump_key in p[0] for p in all_predictions):
                pump_preds = [p[0][pump_key] * p[1] for p in all_predictions]
                ensemble_pred[pump_key] = np.sum(pump_preds, axis=0)

            # Quantiles (if present)
            sample_pred_keys = list(all_predictions[0][0].keys())
            q_keys = [k for k in sample_pred_keys if k.startswith('price_p') and k.endswith(f'_{horizon}h')]
            for qk in q_keys:
                if all(qk in p[0] for p in all_predictions):
                    q_preds = [p[0][qk] * p[1] for p in all_predictions]
                    ensemble_pred[qk] = np.sum(q_preds, axis=0)

            # Uncertainty: std across models (price only)
            if use_uncertainty and all(price_key in p[0] for p in all_predictions):
                price_std = np.std([p[0][price_key] for p in all_predictions], axis=0)
                ensemble_pred[f'{price_key}_uncertainty'] = price_std
        
        return ensemble_pred
    
    def save_ensemble(self, base_path: str = 'models/ensemble'):
        """Save all models in ensemble"""
        os.makedirs(base_path, exist_ok=True)
        
        for i, (model, arch) in enumerate(zip(self.models, self.model_architectures)):
            model.save_model_full(f"{base_path}/{arch}")
        
        # Save ensemble metadata
        with open(f"{base_path}/ensemble_metadata.pkl", 'wb') as f:
            pickle.dump({
                'model_weights': self.model_weights,
                'model_architectures': self.model_architectures,
                'sequence_length': self.sequence_length,
                'prediction_horizons': self.prediction_horizons,
                'n_features': self.n_features
            }, f)
        
        logger.info(f"Ensemble saved to {base_path}")


if __name__ == "__main__":
    """Test enhanced predictor"""
    
    logger.info("Testing enhanced predictor...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 50
    sequence_length = 60
    
    # Simulate sequences
    X_test = np.random.randn(n_samples, sequence_length, n_features)
    
    # Test single model
    predictor = EnhancedCryptoPricePredictor(
        sequence_length=sequence_length,
        prediction_horizons=[1, 6, 24],
        n_features=n_features,
        architecture='hybrid'
    )
    
    model = predictor.build_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Test prediction with uncertainty
    predictions = predictor.predict_with_uncertainty(X_test[:10])
    print("\nSample predictions with uncertainty:")
    for key in list(predictions.keys())[:3]:
        print(f"{key}: shape={predictions[key].shape}")
    
    print("\n" + "="*70)
    print("Enhanced predictor test complete!")
    print("="*70)
