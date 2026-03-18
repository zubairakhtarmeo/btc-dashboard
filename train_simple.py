"""
Simple Training Script with Reduced Feature Set
===============================================
Uses only basic features that don't require extensive history
"""

import numpy as np
import pandas as pd
from datetime import datetime
from data_collector import CryptoDataCollector
from enhanced_predictor import EnhancedCryptoPricePredictor
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_simple_features(df):
    """Add basic features that don't require 720+ hours of history"""
    df = df.copy()
    
    # Basic returns (only need 168 hours history max)
    for h in [1, 6, 12, 24, 48, 168]:
        df[f'return_{h}h'] = df['close'].pct_change(h)
        df[f'log_return_{h}h'] = np.log1p(df[f'return_{h}h'])
    
    # Simple moving averages (max 200 periods)
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
    
    # Volatility (max 100 periods)
    for period in [7, 14, 21, 50]:
        df[f'volatility_{period}'] = df['return_1h'].rolling(period).std()
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # RSI (14-period)
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
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # Momentum indicators
    df['momentum_consistency'] = (
        np.sign(df['return_1h']) == np.sign(df['return_6h'])
    ).astype(int)
    
    # Remove rows with NaN (keep at least 250 rows = ~10 days minimal)
    df_clean = df.dropna()
    logger.info(f"Removed {len(df) - len(df_clean)} rows with NaN values")
    logger.info(f"Remaining samples: {len(df_clean)}")
    
    return df_clean

def main():
    print("="*70)
    print("🚀 TRAINING BITCOIN PRICE PREDICTOR (SIMPLIFIED)")
    print("="*70)
    print()
    
    # 1. Collect real data (2000 hours = ~83 days)
    logger.info("📊 Step 1: Collecting real Bitcoin data...")
    collector = CryptoDataCollector()
    data = collector.collect_all_data('bitcoin', hours_back=2000)
    
    price_df = data['price'].copy()
    logger.info(f"✓ Collected {len(price_df)} price records")
    logger.info(f"✓ Date range: {price_df.index.min()} to {price_df.index.max()}")
    logger.info(f"✓ Price range: ${price_df['close'].min():.2f} - ${price_df['close'].max():.2f}")
    print()
    
    # 2. Add simple features (only needs 200 periods max)
    logger.info("🔧 Step 2: Adding simple features...")
    features_df = add_simple_features(price_df)
    
    logger.info(f"✓ Created {features_df.shape[1]} features")
    logger.info(f"✓ Total samples: {len(features_df)}")
    
    if len(features_df) < 500:
        logger.error("❌ Not enough data! Need at least 500 samples.")
        logger.error(f"Only have {len(features_df)} samples after feature engineering.")
        return
    
    print()
    
    # 3. Prepare for training
    logger.info("📦 Step 3: Preparing model...")
    
    # Remove non-numeric columns
    features_df_clean = features_df.select_dtypes(include=[np.number])
    
    # Build predictor
    predictor = EnhancedCryptoPricePredictor(
        sequence_length=60,
        prediction_horizons=[1, 6, 12, 24, 48, 72, 168],  # 1H, 6H, 12H, 24H, 48H, 72H, 7Days
        n_features=features_df_clean.shape[1],  # Include 'close' to anchor absolute price level
        architecture='hybrid',
        dropout_rate=0.3,
        use_mixed_precision=False
    )
    
    model = predictor.build_model()
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    logger.info(f"✓ Input features: {features_df_clean.shape[1]}")
    print()
    
    # 4. Prepare sequences
    logger.info("📦 Step 4: Preparing sequences...")
    X, y_dict = predictor.prepare_sequences_multihorizon(features_df_clean, target_col='close')
    logger.info(f"✓ Created {len(X)} training sequences")
    logger.info(f"✓ Input shape: {X.shape}")
    print()
    
    if len(X) < 100:
        logger.error("❌ Not enough sequences! Need at least 100.")
        return
    
    # 5. Train
    logger.info("🎓 Step 5: Training model on REAL Bitcoin data...")
    logger.info("⏱️  Training for 20 epochs (this will take several minutes)...")
    print()
    
    # Prepare y_train_list
    y_train_list = []
    for horizon in predictor.prediction_horizons:
        y_train_list.extend([
            y_dict[f'price_{horizon}h'],
            y_dict[f'direction_{horizon}h'],
            y_dict[f'volatility_{horizon}h']
        ])
    
    # Train with callbacks
    history = model.fit(
        X, y_train_list,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ],
        verbose=1
    )
    
    print()
    logger.info("✅ Training complete!")
    print()
    
    # 6. Save model
    logger.info("💾 Step 6: Saving trained model...")
    model_dir = 'models/bitcoin_real_simplified'
    predictor.save_model_full(model_dir)
    logger.info(f"✓ Model saved to {model_dir}")
    print()
    
    # 7. Make predictions on latest data
    logger.info("🔮 Step 7: Generating predictions...")
    # Build an inference sequence from the latest available data (not the last training label,
    # which is ~max_horizon hours behind the present).
    features_array = features_df_clean[predictor.feature_names].values
    features_scaled_full = predictor.scaler_X.transform(features_array)
    X_latest = features_scaled_full[-predictor.sequence_length:].reshape(1, predictor.sequence_length, -1)
    predictions = predictor.predict_with_uncertainty(X_latest)

    # Get current price from the latest close
    current_price = float(features_df_clean['close'].iloc[-1])

    # Realized baseline uncertainty: 1h return std over the last 7 days.
    # This avoids wildly inflated values when the volatility head is uncalibrated.
    try:
        realized_std_1h = float(features_df_clean['return_1h'].tail(168).std())
    except Exception:
        realized_std_1h = float('nan')
    
    print()
    print("="*70)
    print("📊 REAL-TIME PREDICTIONS")
    print("="*70)
    print()
    print(f"💰 Current Bitcoin Price: ${current_price:,.2f}")
    print(f"📅 Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    target_type = getattr(predictor, 'target_type', 'price')
    
    for horizon in predictor.prediction_horizons:
        pred_scaled = float(np.asarray(predictions[f'price_{horizon}h'][0]).reshape(-1)[0])
        pred_std_scaled = float(np.asarray(predictions[f'price_{horizon}h_std'][0]).reshape(-1)[0])

        pred_value = float(predictor.scaler_y.inverse_transform([[pred_scaled]])[0][0])
        if target_type == 'return':
            pred_return = pred_value
            pred_price = current_price * (1.0 + pred_return)

            return_std = pred_std_scaled
            if hasattr(predictor.scaler_y, 'scale_'):
                return_std = return_std * float(predictor.scaler_y.scale_[0])
            std_price_mc = abs(current_price * return_std)

            std_price_real = 0.0
            if np.isfinite(realized_std_1h) and realized_std_1h > 0:
                std_price_real = abs(current_price * realized_std_1h * (float(horizon) ** 0.5))

            pred_std = max(std_price_mc, std_price_real)
        else:
            pred_price = pred_value
            pred_std = pred_std_scaled
            if hasattr(predictor.scaler_y, 'scale_'):
                pred_std = pred_std * float(predictor.scaler_y.scale_[0])
        
        direction_probs = predictions[f'direction_{horizon}h'][0]
        direction_labels = ['DOWN', 'NEUTRAL', 'UP']
        direction = direction_labels[np.argmax(direction_probs)]
        confidence = np.max(direction_probs) * 100
        
        change_pct = ((pred_price - current_price) / current_price * 100)
        
        print(f"🕐 {horizon}-Hour Prediction:")
        print(f"   Price:       ${pred_price:,.2f} ± ${pred_std:,.2f}")
        print(f"   Change:      {change_pct:+.2f}%")
        print(f"   Direction:   {direction} ({confidence:.1f}% confidence)")
        
        # Trading signal
        if direction == 'UP' and confidence > 70:
            signal = '🟢 STRONG BUY'
        elif direction == 'UP' and confidence > 60:
            signal = '🟢 BUY'
        elif direction == 'DOWN' and confidence > 70:
            signal = '🔴 STRONG SELL'
        elif direction == 'DOWN' and confidence > 60:
            signal = '🔴 SELL'
        else:
            signal = '⚪ HOLD'
        
        print(f"   Signal:      {signal}")
        print()
    
    print("="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print()
    print("📈 Training Performance:")
    print(f"   Total epochs:         {len(history.history['loss'])}")
    print(f"   Final training loss:  {history.history['loss'][-1]:.4f}")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print()
    print("💡 Model Details:")
    print(f"   Total parameters:     {model.count_params():,}")
    print(f"   Training sequences:   {len(X):,}")
    print(f"   Features used:        {features_df_clean.shape[1]}")
    print(f"   Prediction horizons:  {', '.join(f'{h}h' for h in predictor.prediction_horizons)}")
    print()
    print("🎯 Next Steps:")
    print(f"   Load model:  predictor.load_model('{model_dir}')")
    print("   Use in production for real-time trading signals")
    print()
    
    return predictor, model, history

if __name__ == "__main__":
    predictor, model, history = main()
