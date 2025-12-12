"""
Real-Time Bitcoin Price Predictor with Live API
Fetches current BTC price from APIs and makes predictions
"""

import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Define absolute paths for model files
MODELS_DIR = project_root / 'models'
MODEL_PATH = MODELS_DIR / 'bitcoin_real_simplified_model.h5'
METADATA_PATH = MODELS_DIR / 'bitcoin_real_simplified_metadata.pkl'

from enhanced_predictor import EnhancedCryptoPricePredictor
from data_collector import CryptoDataCollector

def add_simple_features(df):
    """Add basic features matching train_simple.py exactly"""
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
    
    df_clean = df.dropna()
    return df_clean

def predict_live():
    """
    Make predictions using REAL-TIME BTC price from live APIs
    """
    
    print("\n" + "="*70)
    print(" REAL-TIME BITCOIN PRICE PREDICTOR")
    print("="*70)
    print(f"\n Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if model files exist
    if not MODEL_PATH.exists():
        print(f"\n Model not found at: {MODEL_PATH}")
        print(" Please run 'python train_simple.py' first to train the model")
        return
    
    if not METADATA_PATH.exists():
        print(f"\n Metadata not found at: {METADATA_PATH}")
        print(" Please run 'python train_simple.py' first to train the model")
        return
    
    # Initialize collector
    logger.info("\n Fetching LIVE Bitcoin price from APIs...")
    collector = CryptoDataCollector(use_cache=True)
    
    try:
        # Get REAL-TIME current price from live APIs
        current_price = collector.get_current_price('bitcoin')
        
        print(f"\n Current BTC Price (LIVE API): ${current_price:,.2f}")
        
        # Get historical data for context
        logger.info(" Loading historical data for context...")
        price_data = collector.price_collector.get_price_data('bitcoin', hours_back=2000, interval='1h')
        
        # Override the LAST price with live current price
        price_data.iloc[-1, price_data.columns.get_loc('close')] = current_price
        
        logger.info(f" Loaded {len(price_data)} historical records")
        logger.info(f" Updated latest price to LIVE value: ${current_price:,.2f}")
        
    except Exception as e:
        logger.error(f" Failed to fetch live price: {e}")
        logger.error(" Make sure you have internet connection")
        return
    
    # Load metadata first for feature names
    import pickle
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    # Generate features using the SAME method as training
    features_df = add_simple_features(price_data)
    
    # Remove non-numeric columns and select only model features
    features_df_clean = features_df.select_dtypes(include=[np.number])
    feature_names = metadata['feature_names']
    features = features_df_clean[feature_names]
    
    logger.info(f" Generated {len(features.columns)} features (matching model training)")
    
    # Load trained model
    logger.info(" Loading trained model...")
    
    import keras
    
    # Import custom layers
    from enhanced_predictor import TemporalConvLayer, MultiHeadAttentionCustom
    
    # Load model with custom objects and compile=False to avoid metric issues
    custom_objects = {
        'TemporalConvLayer': TemporalConvLayer,
        'MultiHeadAttentionCustom': MultiHeadAttentionCustom,
        'mse': keras.losses.MeanSquaredError()
    }
    model = keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objects, compile=False)
    
    config = metadata['config']
    predictor = EnhancedCryptoPricePredictor(
        sequence_length=config['sequence_length'],
        n_features=config['n_features'],
        prediction_horizons=config['prediction_horizons']
    )
    predictor.model = model
    predictor.feature_scaler = metadata['scaler_X']
    predictor.price_scaler = metadata['scaler_y']
    
    logger.info(f" Loaded model from {MODEL_PATH}")
    
    # Make predictions
    logger.info("\n Generating predictions...\n")
    
    # Get feature names (exclude 'close' as it's the target)
    feature_cols = [col for col in features.columns if col != 'close']
    
    # Scale features using loaded scaler
    features_array = features[feature_cols].values
    features_scaled = predictor.feature_scaler.transform(features_array)
    
    # Create sequence (take last sequence_length rows)
    sequence_length = config['sequence_length']
    if len(features_scaled) < sequence_length:
        logger.error(f" Not enough data! Need {sequence_length} rows, have {len(features_scaled)}")
        return
    
    # Take the last sequence_length rows and reshape to (1, sequence_length, n_features)
    X = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
    logger.info(f" Created prediction sequence: shape {X.shape}")
    
    predictions = predictor.predict_with_uncertainty(X)
    
    # Display results
    print("\n" + "="*70)
    print(" PREDICTIONS")
    print("="*70 + "\n")
    
    # Prepare data for Excel export
    excel_data = []
    
    horizons = [1, 6, 12, 24, 48, 72, 168]
    horizon_labels = ['1h', '6h', '12h', '24h', '48h', '72h', '7days']
    
    for horizon, label in zip(horizons, horizon_labels):
        # Get predictions (they are in scaled space)
        pred_scaled = predictions[f'price_{horizon}h'][0]
        pred_std_scaled = predictions[f'price_{horizon}h_std'][0]
        direction_prob = predictions[f'direction_{horizon}h'][0]  # [down, neutral, up]
        direction_std = predictions[f'direction_{horizon}h_std'][0]
        
        # Ensure pred_scaled is a scalar
        if isinstance(pred_scaled, np.ndarray):
            pred_scaled = float(pred_scaled.flatten()[0])
        
        # Inverse transform price predictions to original scale
        pred_price = predictor.price_scaler.inverse_transform([[pred_scaled]])[0, 0]
        # Scale std by the scaler's scale factor
        if hasattr(predictor.price_scaler, 'scale_'):
            pred_std_price = float(pred_std_scaled.flatten()[0]) * predictor.price_scaler.scale_[0]
        else:
            pred_std_price = float(pred_std_scaled.flatten()[0]) * 1000  # Fallback estimate
        
        # Calculate change
        change_pct = ((pred_price - current_price) / current_price) * 100
        
        # Interpret direction probabilities: [down, neutral, up]
        prob_down = direction_prob[0]
        prob_neutral = direction_prob[1]
        prob_up = direction_prob[2]
        
        # Determine direction and confidence
        max_prob = max(prob_down, prob_neutral, prob_up)
        if prob_up > prob_down and prob_up > prob_neutral:
            direction_str = "UP"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction_str = "DOWN"
            confidence = prob_down
        else:
            direction_str = "NEUTRAL"
            confidence = prob_neutral
        
        # Determine signal
        if confidence > 0.5 and max_prob > 0.6:
            if direction_str == "UP":
                signal = "BUY"
            elif direction_str == "DOWN":
                signal = "SELL"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
        
        # Store for Excel
        excel_data.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Horizon': label.upper(),
            'Current_Price': current_price,
            'Predicted_Price': pred_price,
            'Price_Change_USD': pred_price - current_price,
            'Price_Change_Percent': change_pct,
            'Uncertainty': pred_std_price,
            'Direction': direction_str,
            'Confidence_Percent': confidence * 100,
            'Prob_Down_Percent': prob_down * 100,
            'Prob_Neutral_Percent': prob_neutral * 100,
            'Prob_Up_Percent': prob_up * 100,
            'Signal': signal
        })
        
        print(f" {label.upper()} Prediction:")
        print(f"   Price:       ${pred_price:,.2f} ± ${pred_std_price:,.2f}")
        print(f"   Change:      {change_pct:+.2f}%")
        print(f"   Direction:   {direction_str} ({confidence*100:.1f}% confidence)")
        print(f"   Probabilities: ↓{prob_down*100:.1f}% ↔{prob_neutral*100:.1f}% ↑{prob_up*100:.1f}%")
        print(f"   Signal:      {signal}")
        print()
    
    # Save to Excel
    output_dir = Path("D:/apparel/crypto")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = output_dir / f"btc_prediction_{timestamp}.xlsx"
    
    df_results = pd.DataFrame(excel_data)
    
    # Create Excel writer with formatting
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Get the worksheet
        worksheet = writer.sheets['Predictions']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print("="*70)
    print(" Prediction complete!")
    print(f" Results saved to: {excel_filename}")
    print("="*70 + "\n")
    
    return excel_filename

def main():
    """Main entry point"""
    
    try:
        predict_live()
        
    except KeyboardInterrupt:
        print("\n\n Prediction cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
