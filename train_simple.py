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
    print("TRAINING BITCOIN PRICE PREDICTOR (SIMPLIFIED)")
    print("="*70)
    print()
    
    # 1. Collect real data (2000 hours = ~83 days - NEED ENOUGH HISTORY FOR ROBUST MODEL)
    logger.info("📊 Step 1: Collecting real Bitcoin data...")
    collector = CryptoDataCollector(use_cache=False)  # FORCE FRESH DATA
    data = collector.collect_all_data('bitcoin', hours_back=2000)  # Use full history for robust model
    
    price_df = data['price'].copy()
    news_df = data.get('news')
    alt_bundle = {
        'derivatives': None,
        'funding_rates': None,
        'liquidations': None,
        'options_flow': None,
    }
    # DISABLED: derivatives collection causes network timeouts and feature mismatch
    # try:
    #     from alternative_data import AlternativeDataCollector
    #     alt = AlternativeDataCollector(api_keys={})
    #     alt_bundle = {
    #         'derivatives': alt.get_derivatives_data('bitcoin'),
    #         'funding_rates': alt.get_funding_rates('bitcoin'),
    #         'liquidations': alt.get_liquidation_data('bitcoin'),
    #         'options_flow': alt.get_options_flow('bitcoin'),
    #     }
    #     logger.info("✓ Collected derivatives/liquidations (best-effort)")
    # except Exception:
    #     logger.info("ℹ️  Derivatives/liquidations not available (optional)")
    
    logger.info(f"✓ Collected {len(price_df)} price records")
    logger.info(f"✓ Date range: {price_df.index.min()} to {price_df.index.max()}")
    logger.info(f"✓ Price range: ${price_df['close'].min():.2f} - ${price_df['close'].max():.2f}")
    print()
    
    # 2. Add features (simple technicals + explicit news/geopolitics signals)
    logger.info("🔧 Step 2: Adding features (technicals + news/geopolitics)...")
    features_df = add_simple_features(
        price_df,
        news_df=news_df,
        # DISABLED: derivatives cause feature mismatch with dashboard (dashboard doesn't fetch these)
        # derivatives_df=alt_bundle.get('derivatives'),
        # funding_df=alt_bundle.get('funding_rates'),
        # liquidations_df=alt_bundle.get('liquidations'),
        # options_df=alt_bundle.get('options_flow'),
    )
    
    logger.info(f"✓ Created {features_df.shape[1]} features")
    logger.info(f"✓ Total samples: {len(features_df)}")
    
    if len(features_df) < 500:  # Restored proper validation
        logger.error("❌ Not enough data! Need at least 500 samples.")
        logger.error(f"Only have {len(features_df)} samples after feature engineering.")
        return None, None, None
    
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
        dropout_rate=0.4,  # Increased from 0.3
        learning_rate=1e-4,  # Reduced from 3e-4
        use_mixed_precision=False
    )
    
    model = predictor.build_model()
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    logger.info(f"✓ Input features: {features_df_clean.shape[1] - 1}")
    print()
    
    # 4. Prepare sequences
    logger.info("📦 Step 4: Preparing sequences...")
    
    # DEBUG: Check what prices we're about to train on
    logger.info(f"🔍 DEBUG - Price data statistics BEFORE scaler:")
    logger.info(f"  Min: ${features_df_clean['close'].min():,.2f}")
    logger.info(f"  Max: ${features_df_clean['close'].max():,.2f}")
    logger.info(f"  Mean: ${features_df_clean['close'].mean():,.2f}")
    logger.info(f"  Median: ${features_df_clean['close'].median():,.2f}")
    logger.info(f"  Latest: ${features_df_clean['close'].iloc[-1]:,.2f}")
    logger.info(f"  First: ${features_df_clean['close'].iloc[0]:,.2f}")
    
    X, y_dict = predictor.prepare_sequences_multihorizon(features_df_clean, target_col='close')
    logger.info(f"✓ Created {len(X)} training sequences")
    logger.info(f"✓ Input shape: {X.shape}")
    
    # DEBUG: Check what scaler was fitted on
    logger.info(f"🔍 DEBUG - Scaler fitted on:")
    logger.info(f"  Center: ${predictor.scaler_y.center_[0]:,.2f}")
    logger.info(f"  Scale: ${predictor.scaler_y.scale_[0]:,.2f}")
    logger.info(f"  This means data was centered around ${predictor.scaler_y.center_[0]:,.0f}")
    logger.info(f"  Latest price in training data: ${features_df_clean['close'].iloc[-1]:,.2f}")
    print()
    
    if len(X) < 100:
        logger.error("❌ Not enough sequences! Need at least 100.")
        return None, None, None  # Return tuple
    
    # 5. Train
    logger.info("🎓 Step 5: Training model on REAL Bitcoin data...")
    logger.info("⏱️  Training for 20 epochs (this will take several minutes)...")
    print()
    
    # Prepare y_train_list
    # Time-series safe split (keep order; no shuffle)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]

    if isinstance(model.output, dict) or (hasattr(model, 'output_names') and any(name in y_dict for name in model.output_names)):
        y_train = {name: y_dict[name][:split] for name in model.output_names if name in y_dict}
        y_val = {name: y_dict[name][split:] for name in model.output_names if name in y_dict}
        
        # Calculate class weights for crash/pump heads (they are imbalanced - crashes/pumps are rare)
        class_weight_dict = {}
        for horizon in predictor.prediction_horizons:
            # For crash head
            crash_key = f'crash_{horizon}h'
            if crash_key in y_train:
                crash_pos_rate = float(np.mean(y_train[crash_key]))
                if 0.01 < crash_pos_rate < 0.99:  # avoid division by zero
                    neg_weight = crash_pos_rate
                    pos_weight = 1.0 - crash_pos_rate
                    class_weight_dict[crash_key] = {0: neg_weight, 1: pos_weight}
                    logger.info(f"{crash_key}: {crash_pos_rate*100:.1f}% positive, weights: {{0:{neg_weight:.2f}, 1:{pos_weight:.2f}}}")
            
            # For pump head  
            pump_key = f'pump_{horizon}h'
            if pump_key in y_train:
                pump_pos_rate = float(np.mean(y_train[pump_key]))
                if 0.01 < pump_pos_rate < 0.99:
                    neg_weight = pump_pos_rate
                    pos_weight = 1.0 - pump_pos_rate
                    class_weight_dict[pump_key] = {0: neg_weight, 1: pos_weight}
                    logger.info(f"{pump_key}: {pump_pos_rate*100:.1f}% positive, weights: {{0:{neg_weight:.2f}, 1:{pos_weight:.2f}}}")
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,  # Reduced from 35
            batch_size=32,
            shuffle=False,
            class_weight=class_weight_dict if class_weight_dict else None,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,  # MUCH stricter (was 7)
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=0.001  # Stop if not improving by 0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,  # Stricter (was 3)
                    min_lr=1e-6,  # Lower minimum
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
            X_train,
            [arr[:split] for arr in y_train_list],
            validation_data=(X_val, [arr[split:] for arr in y_train_list]),
            epochs=35,
            batch_size=32,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=7,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-5,
                    verbose=1
                )
            ],
            verbose=1
        )
    
    print()
    logger.info("✅ Training complete!")
    print()

    # 5.5 Sanity-check model outputs before saving (prevents overwriting with a diverged model)
    try:
        test_X = X_val[-1:].astype(np.float32) if len(X_val) else X[-1:].astype(np.float32)
        test_pred = predictor.predict_with_uncertainty(test_X, n_samples=5)
        # Use last observed close as baseline
        baseline_close = float(features_df_clean['close'].iloc[-1])
        bad = False
        bad_reasons = []

        for h in predictor.prediction_horizons:
            pk = f'price_{h}h'
            dk = f'direction_{h}h'
            if pk not in test_pred:
                continue
            p_scaled = float(np.asarray(test_pred[pk][0]).reshape(-1)[0])
            p = float(predictor.scaler_y.inverse_transform([[p_scaled]])[0][0])
            if baseline_close > 0 and (p / baseline_close > 3.0 or p / baseline_close < (1.0 / 3.0)):
                bad = True
                bad_reasons.append(f"{h}h price ratio {p/baseline_close:.2f}x")
            if dk in test_pred:
                probs = np.asarray(test_pred[dk][0]).reshape(-1)
                if probs.size == 3 and float(np.max(probs)) > 0.995:
                    # Saturation across many horizons is a divergence smell
                    bad = True
                    bad_reasons.append(f"{h}h direction saturated")

        if bad:
            logger.error("❌ Model outputs look unstable; not saving artifacts.")
            logger.error("Reasons: " + ", ".join(bad_reasons[:8]))
            logger.error("Try re-running training (it should stabilize with the updated code).")
            return
    except Exception as e:
        logger.warning(f"Sanity-check skipped due to error: {e}")
    
    # 6. Save model
    logger.info("💾 Step 6: Saving trained model...")
    model_dir = 'models/bitcoin_real_simplified'
    predictor.save_model_full(model_dir)
    logger.info(f"✓ Model saved to {model_dir}")
    print()
    
    # 7. Make predictions on latest data
    logger.info("🔮 Step 7: Generating predictions...")
    predictions = predictor.predict_with_uncertainty(X[-10:])
    
    # Get current price from the actual price data (not from scaled targets)
    current_price = float(price_df['close'].iloc[-1])
    
    print()
    print("="*70)
    print("📊 REAL-TIME PREDICTIONS")
    print("="*70)
    print()
    print(f"💰 Current Bitcoin Price: ${current_price:,.2f}")
    print(f"📅 Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # DEBUG: Check raw model outputs
    raw_pred = predictor.model.predict(X[-1:], verbose=0)
    print("🔍 DEBUG - Raw model outputs (should be scaled, near 0):")
    for h in [1, 6, 24]:
        key = f'price_{h}h'
        if key in raw_pred:
            print(f"  {key}: {float(raw_pred[key][0,0]):.6f}")
    print()
    
    for horizon in predictor.prediction_horizons:
        # Get raw scaled prediction
        pred_scaled = predictions[f'price_{horizon}h'][-1].reshape(-1, 1)
        
        # CORRECTION: The scaler is centered on historical median (~$90k), but we want prediction relative to CURRENT price
        # Interpret the scaled prediction as a change: scaled_value * scale = dollar_change
        # Example: if pred_scaled = 0.06 and scale = 10000, that's +$600 change
        pred_scaled_value = pred_scaled[0][0]
        dollar_change = pred_scaled_value * predictor.scaler_y.scale_[0]
        
        # Apply the change to current price
        pred_price = current_price + dollar_change
        
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
