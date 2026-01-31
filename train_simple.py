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

from simple_features import add_simple_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    news_df = data.get('news')
    alt_bundle = {
        'derivatives': None,
        'funding_rates': None,
        'liquidations': None,
        'options_flow': None,
    }
    try:
        from alternative_data import AlternativeDataCollector

        alt = AlternativeDataCollector(api_keys={})
        alt_bundle = {
            'derivatives': alt.get_derivatives_data('bitcoin'),
            'funding_rates': alt.get_funding_rates('bitcoin'),
            'liquidations': alt.get_liquidation_data('bitcoin'),
            'options_flow': alt.get_options_flow('bitcoin'),
        }
        logger.info("✓ Collected derivatives/liquidations (best-effort)")
    except Exception:
        logger.info("ℹ️  Derivatives/liquidations not available (optional)")
    logger.info(f"✓ Collected {len(price_df)} price records")
    logger.info(f"✓ Date range: {price_df.index.min()} to {price_df.index.max()}")
    logger.info(f"✓ Price range: ${price_df['close'].min():.2f} - ${price_df['close'].max():.2f}")
    print()
    
    # 2. Add features (simple technicals + explicit news/geopolitics signals)
    logger.info("🔧 Step 2: Adding features (technicals + news/geopolitics)...")
    features_df = add_simple_features(
        price_df,
        news_df=news_df,
        derivatives_df=alt_bundle.get('derivatives'),
        funding_df=alt_bundle.get('funding_rates'),
        liquidations_df=alt_bundle.get('liquidations'),
        options_df=alt_bundle.get('options_flow'),
    )
    
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
        n_features=features_df_clean.shape[1] - 1,  # Exclude 'close'
        architecture='hybrid',
        dropout_rate=0.3,
        learning_rate=3e-4,
        use_mixed_precision=False
    )
    
    model = predictor.build_model()
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    logger.info(f"✓ Input features: {features_df_clean.shape[1] - 1}")
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
    if isinstance(model.output, dict) or (hasattr(model, 'output_names') and any(name in y_dict for name in model.output_names)):
        y_train = {name: y_dict[name] for name in model.output_names if name in y_dict}
        history = model.fit(
            X,
            y_train,
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
    else:
        y_train_list = []
        for horizon in predictor.prediction_horizons:
            y_train_list.extend([
                y_dict[f'price_{horizon}h'],
                y_dict[f'direction_{horizon}h'],
                y_dict[f'volatility_{horizon}h']
            ])

        history = model.fit(
            X,
            y_train_list,
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
    predictions = predictor.predict_with_uncertainty(X[-10:])
    
    # Get current price
    current_price = predictor.scaler_y.inverse_transform([[float(y_dict['price_1h'][-1, 0])]])[0][0]
    
    print()
    print("="*70)
    print("📊 REAL-TIME PREDICTIONS")
    print("="*70)
    print()
    print(f"💰 Current Bitcoin Price: ${current_price:,.2f}")
    print(f"📅 Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for horizon in predictor.prediction_horizons:
        pred_price = predictor.scaler_y.inverse_transform(
            predictions[f'price_{horizon}h'][-1].reshape(-1, 1)
        )[0][0]
        pred_std = predictor.scaler_y.scale_[0] * predictions[f'price_{horizon}h_std'][-1][0]
        
        direction_probs = predictions[f'direction_{horizon}h'][-1]
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
    print(f"   Features used:        {features_df_clean.shape[1] - 1}")
    print(f"   Prediction horizons:  1h, 6h, 24h")
    print()
    print("🎯 Next Steps:")
    print(f"   Load model:  predictor.load_model('{model_dir}')")
    print("   Use in production for real-time trading signals")
    print()
    
    return predictor, model, history

if __name__ == "__main__":
    predictor, model, history = main()
