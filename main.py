"""
Main Application - Crypto Price Predictor
==========================================

Orchestrates the entire prediction pipeline:
1. Data collection
2. Feature engineering  
3. Sentiment analysis
4. Model prediction
5. Signal generation

This is the entry point for running the system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import argparse
import json
import os

# Import modules
from config import Config
from data_collector import CryptoDataCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineering import FeatureEngineer
from price_predictor import CryptoPricePredictor, EnsemblePredictor
from enhanced_predictor import EnhancedCryptoPricePredictor, HeterogeneousEnsemble
from signal_generator import SignalGenerator
from backtester import Backtester

# Setup logging
os.makedirs('logs', exist_ok=True)
handler = RotatingFileHandler(
    Config.LOGGING['file'],
    maxBytes=Config.LOGGING['max_bytes'],
    backupCount=Config.LOGGING['backup_count']
)
handler.setFormatter(logging.Formatter(Config.LOGGING['format']))

logging.basicConfig(
    level=getattr(logging, Config.LOGGING['level']),
    format=Config.LOGGING['format'],
    handlers=[handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class CryptoPredictionSystem:
    """
    Complete cryptocurrency price prediction and trading signal system
    """
    
    def __init__(self, symbol: str = 'bitcoin', train_model: bool = False, use_enhanced: bool = True, backtest_mode: bool = False):
        """
        Initialize the system
        
        Args:
            symbol: Cryptocurrency to analyze
            train_model: Whether to train a new model
            use_enhanced: Use enhanced predictor with advanced features
            backtest_mode: Run in backtesting mode
        """
        self.symbol = symbol
        self.train_model = train_model
        self.use_enhanced = use_enhanced
        self.backtest_mode = backtest_mode
        
        logger.info("="*70)
        logger.info("CRYPTO PRICE PREDICTOR - AI/ML Trading System")
        logger.info("="*70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info(f"Mode: {'Enhanced' if use_enhanced else 'Standard'}")
        logger.info(f"Backtesting: {'Enabled' if backtest_mode else 'Disabled'}")
        
        # Initialize components
        self.data_collector = CryptoDataCollector(Config.API_KEYS)
        self.sentiment_analyzer = SentimentAnalyzer(
            use_transformers=Config.SENTIMENT['use_transformers']
        )
        self.feature_engineer = FeatureEngineer()
        
        # Select predictor type
        if self.use_enhanced:
            if Config.ENSEMBLE['use_ensemble']:
                self.predictor = HeterogeneousEnsemble(
                    sequence_length=Config.MODEL['sequence_length'],
                    prediction_horizons=Config.MODEL.get('prediction_horizons', [1, 6, 24])
                )
            else:
                self.predictor = EnhancedCryptoPricePredictor(
                    sequence_length=Config.MODEL['sequence_length'],
                    prediction_horizons=Config.MODEL.get('prediction_horizons', [1, 6, 24]),
                    architecture=Config.MODEL.get('architecture', 'hybrid'),
                    use_mixed_precision=Config.MODEL.get('use_mixed_precision', True)
                )
        else:
            if Config.ENSEMBLE['use_ensemble']:
                self.predictor = EnsemblePredictor(n_models=Config.ENSEMBLE['n_models'])
            else:
                self.predictor = CryptoPricePredictor(**Config.MODEL)
        
        self.signal_generator = SignalGenerator(
            risk_tolerance=Config.SIGNALS['risk_tolerance'],
            min_confidence=Config.SIGNALS['min_confidence']
        )
        
        # Initialize backtester if needed
        self.backtester = None
        if self.backtest_mode:
            self.backtester = Backtester(
                initial_capital=Config.BACKTESTING.get('initial_capital', 10000.0),
                position_size=Config.BACKTESTING.get('position_size', 0.1),
                transaction_cost=Config.BACKTESTING.get('transaction_cost', 0.001)
            )
        
        logger.info("System components initialized successfully")
    
    def collect_data(self) -> dict:
        """Collect all required data"""
        logger.info("\n" + "-"*70)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("-"*70)
        
        data = self.data_collector.collect_all_data(
            symbol=self.symbol,
            hours_back=Config.DATA_COLLECTION['historical_hours']
        )
        
        logger.info("✓ Data collection complete")
        return data
    
    def analyze_sentiment(self, data: dict) -> dict:
        """Analyze sentiment from news and social media"""
        logger.info("\n" + "-"*70)
        logger.info("STEP 2: SENTIMENT ANALYSIS")
        logger.info("-"*70)
        
        # Analyze news
        if data['news'] is not None and not data['news'].empty:
            data['news'] = self.sentiment_analyzer.analyze_news(data['news'])
        
        # Analyze social media
        if data['social'] is not None and not data['social'].empty:
            data['social'] = self.sentiment_analyzer.analyze_social_media(data['social'])
        
        # Get sentiment indicators
        sentiment_indicators = self.sentiment_analyzer.get_sentiment_indicators(
            data['news'], data['social']
        )
        
        logger.info("✓ Sentiment analysis complete")
        logger.info(f"  Combined sentiment: {sentiment_indicators['combined_sentiment']:.3f}")
        
        return data
    
    def engineer_features(self, data: dict) -> pd.DataFrame:
        """Engineer features from all data sources"""
        logger.info("\n" + "-"*70)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("-"*70)
        
        features_df = self.feature_engineer.engineer_all_features(
            price_df=data['price'],
            news_df=data['news'],
            social_df=data['social'],
            onchain_df=data['onchain'],
            market_df=data['market']
        )
        
        logger.info("✓ Feature engineering complete")
        logger.info(f"  Total features: {len(features_df.columns)}")
        logger.info(f"  Total samples: {len(features_df)}")
        
        return features_df
    
    def train_or_load_model(self, features_df: pd.DataFrame):
        """Train new model or load existing one"""
        logger.info("\n" + "-"*70)
        logger.info("STEP 4: MODEL PREPARATION")
        logger.info("-"*70)
        
        model_path = f"models/{self.symbol}_{'enhanced' if self.use_enhanced else 'standard'}_predictor.h5"
        
        if self.train_model or not os.path.exists(model_path):
            logger.info("Training new model...")
            
            if self.use_enhanced:
                # Enhanced predictor with multi-horizon targets
                if isinstance(self.predictor, HeterogeneousEnsemble):
                    # Build ensemble first
                    self.predictor.build_ensemble()
                    # Use first model for sequence preparation
                    X, y_dict = self.predictor.models[0].prepare_sequences_multihorizon(
                        features_df, target_col='close'
                    )
                    # Train ensemble
                    self.predictor.train_ensemble_parallel(
                        X, y_dict,
                        use_time_series_cv=Config.ENSEMBLE.get('use_time_series_cv', True),
                        epochs=Config.MODEL['epochs'],
                        batch_size=Config.MODEL['batch_size']
                    )
                else:
                    X, y_dict = self.predictor.prepare_sequences_multihorizon(
                        features_df, target_col='close'
                    )
                    self.predictor.train(
                        X, list(y_dict.values()),
                        epochs=Config.MODEL['epochs'],
                        batch_size=Config.MODEL['batch_size']
                    )
                    self.predictor.save_model_full(model_path.replace('.h5', ''))
            else:
                # Standard predictor
                X, y_price, y_direction, y_volatility = self.predictor.models[0].prepare_sequences(
                    features_df, target_col='close'
                ) if Config.ENSEMBLE['use_ensemble'] else self.predictor.prepare_sequences(
                    features_df, target_col='close'
                )
                
                if Config.ENSEMBLE['use_ensemble']:
                    self.predictor.build_ensemble(**Config.MODEL)
                    self.predictor.train_ensemble(
                        X, y_price, y_direction, y_volatility,
                        epochs=Config.MODEL['epochs'],
                        batch_size=Config.MODEL['batch_size']
                    )
                else:
                    self.predictor.train(
                        X, y_price, y_direction, y_volatility,
                        epochs=Config.MODEL['epochs'],
                        batch_size=Config.MODEL['batch_size']
                    )
                    self.predictor.save_model(model_path)
            
            logger.info("✓ Model training complete")
        else:
            logger.info(f"Loading existing model from {model_path}")
            if self.use_enhanced and not isinstance(self.predictor, HeterogeneousEnsemble):
                self.predictor.load_model_full(model_path.replace('.h5', ''))
            elif not Config.ENSEMBLE['use_ensemble']:
                self.predictor.load_model(model_path)
            logger.info("✓ Model loaded")
    
    def make_prediction(self, features_df: pd.DataFrame) -> dict:
        """Make price predictions"""
        logger.info("\n" + "-"*70)
        logger.info("STEP 5: PREDICTION")
        logger.info("-"*70)
        
        # Prepare latest sequence
        if self.use_enhanced:
            if isinstance(self.predictor, HeterogeneousEnsemble):
                X, _ = self.predictor.lstm_model.prepare_sequences_multihorizon(
                    features_df, target_col='close'
                )
                predictions = self.predictor.predict_ensemble(X[-1:], use_uncertainty=True)
            else:
                X, _ = self.predictor.prepare_sequences_multihorizon(features_df, target_col='close')
                predictions = self.predictor.predict_with_uncertainty(X[-1:])
            
            logger.info("✓ Enhanced prediction complete")
            logger.info(f"  Predicted price (1h):  ${predictions.get('price_1h', [0])[-1]:,.2f}")
            logger.info(f"  Predicted price (6h):  ${predictions.get('price_6h', [0])[-1]:,.2f}")
            logger.info(f"  Predicted price (24h): ${predictions.get('price_24h', [0])[-1]:,.2f}")
            if 'price_24h_std' in predictions:
                logger.info(f"  Uncertainty (24h): ±${predictions['price_24h_std'][-1]:,.2f}")
        else:
            if Config.ENSEMBLE['use_ensemble']:
                X, _, _, _ = self.predictor.models[0].prepare_sequences(features_df, target_col='close')
            else:
                X, _, _, _ = self.predictor.prepare_sequences(features_df, target_col='close')
            
            predictions = self.predictor.predict(X[-1:])
            
            logger.info("✓ Prediction complete")
            logger.info(f"  Predicted price: ${predictions['price'][-1]:,.2f}")
            logger.info(f"  Direction: {['Down', 'Neutral', 'Up'][predictions['direction'][-1]]}")
            logger.info(f"  Confidence: {predictions['direction_confidence'][-1]:.1%}")
        
        return predictions
    
    def generate_trading_signal(
        self,
        predictions: dict,
        features_df: pd.DataFrame,
        current_price: float
    ):
        """Generate trading signals"""
        logger.info("\n" + "-"*70)
        logger.info("STEP 6: SIGNAL GENERATION")
        logger.info("-"*70)
        
        # Convert predictions to format expected by signal generator
        if self.use_enhanced:
            # Use 24h prediction as primary
            ml_prediction = {
                'price': predictions.get('price_24h', predictions.get('price', [current_price]))[-1],
                'direction': predictions.get('direction_24h', predictions.get('direction', [1]))[-1],
                'direction_confidence': predictions.get('direction_confidence_24h', 
                                                       predictions.get('direction_confidence', [0.5]))[-1],
                'volatility': predictions.get('volatility', [0.02])[-1] if 'volatility' in predictions 
                            else 0.02,
                # Add uncertainty if available
                'uncertainty': predictions.get('price_24h_std', [0])[-1] if 'price_24h_std' in predictions 
                             else None
            }
        else:
            ml_prediction = {
                'price': predictions['price'][-1],
                'direction': predictions['direction'][-1],
                'direction_confidence': predictions['direction_confidence'][-1],
                'volatility': predictions['volatility'][-1]
            }
        
        # Generate signal
        signal = self.signal_generator.generate_signal(
            ml_prediction, features_df, current_price
        )
        
        logger.info("✓ Signal generation complete")
        
        return signal
    
    def run(self):
        """Run complete prediction pipeline"""
        try:
            # Check if backtesting mode
            if self.backtest_mode:
                return self.run_backtest()
            
            # 1. Collect data
            data = self.collect_data()
            
            # 2. Analyze sentiment
            data = self.analyze_sentiment(data)
            
            # 3. Engineer features
            features_df = self.engineer_features(data)
            
            # 4. Train or load model
            self.train_or_load_model(features_df)
            
            # 5. Make prediction
            predictions = self.make_prediction(features_df)
            
            # 6. Generate signal
            current_price = features_df['close'].iloc[-1]
            signal = self.generate_trading_signal(predictions, features_df, current_price)
            
            # 7. Display results
            self.display_results(signal)
            
            # 8. Save results
            self.save_results(signal)
            
            logger.info("\n" + "="*70)
            logger.info("SYSTEM EXECUTION COMPLETE")
            logger.info("="*70)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in system execution: {e}", exc_info=True)
            raise
    
    def run_backtest(self):
        """Run backtesting mode"""
        logger.info("\n" + "="*70)
        logger.info("BACKTESTING MODE")
        logger.info("="*70)
        
        try:
            # 1. Collect extended historical data
            logger.info("Collecting extended historical data for backtesting...")
            data = self.data_collector.collect_all_data(
                symbol=self.symbol,
                hours_back=Config.BACKTESTING.get('walk_forward_window', 720) + 720  # Extra data for training
            )
            
            # 2. Analyze sentiment
            data = self.analyze_sentiment(data)
            
            # 3. Engineer features
            features_df = self.engineer_features(data)
            
            logger.info(f"Total data points for backtesting: {len(features_df)}")
            
            # 4. Run backtest
            logger.info("\nRunning walk-forward backtesting...")
            results = self.backtester.run_backtest(
                features_df=features_df,
                predictor=self.predictor,
                signal_generator=self.signal_generator,
                walk_forward_window=Config.BACKTESTING.get('walk_forward_window', 720),
                use_enhanced=self.use_enhanced
            )
            
            # 5. Display results
            self.display_backtest_results(results)
            
            # 6. Plot results
            logger.info("\nGenerating backtest plots...")
            self.backtester.plot_results(results)
            
            # 7. Save results
            os.makedirs('backtest_results', exist_ok=True)
            filename = f"backtest_results/{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filename}")
            logger.info("\n" + "="*70)
            logger.info("BACKTESTING COMPLETE")
            logger.info("="*70)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}", exc_info=True)
            raise
    
    def display_backtest_results(self, results: dict):
        """Display backtesting results"""
        print("\n" + "="*70)
        print("BACKTESTING RESULTS")
        print("="*70)
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total Return:     {results['total_return']:>12.2f}%")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown']:>12.2f}%")
        print(f"  Win Rate:         {results['win_rate']:>12.2f}%")
        print(f"  Total Trades:     {results['total_trades']:>12}")
        
        print(f"\nCAPITAL:")
        print(f"  Initial:          ${results['initial_capital']:>11,.2f}")
        print(f"  Final:            ${results['final_capital']:>11,.2f}")
        print(f"  Peak:             ${results['peak_capital']:>11,.2f}")
        
        print(f"\nCOMPARISON:")
        buy_hold_return = results.get('buy_hold_return', 0)
        excess_return = results['total_return'] - buy_hold_return
        print(f"  Buy & Hold:       {buy_hold_return:>12.2f}%")
        print(f"  Excess Return:    {excess_return:>12.2f}%")
        
        print("\n" + "="*70)
    
    def display_results(self, signal):
        """Display trading signal in formatted output"""
        print("\n" + "="*70)
        print("TRADING SIGNAL RECOMMENDATION")
        print("="*70)
        print(f"\nSymbol: {self.symbol.upper()}")
        print(f"Timestamp: {signal.timestamp}")
        print(f"\n{'='*70}")
        print(f"SIGNAL: {signal.signal_type.value}")
        print(f"Confidence: {signal.confidence:.1%}")
        print(f"{'='*70}")
        
        print(f"\nPRICE ANALYSIS:")
        print(f"  Current Price:    ${signal.current_price:>12,.2f}")
        print(f"  Predicted Price:  ${signal.predicted_price:>12,.2f}")
        print(f"  Expected Return:  {signal.expected_return:>12.2f}%")
        
        print(f"\nRISK ASSESSMENT:")
        print(f"  Risk Level:       {signal.risk_level.value:>12}")
        print(f"  Volatility:       {signal.volatility:>12.2%}")
        
        print(f"\nTRADING RECOMMENDATIONS:")
        print(f"  Recommended Entry: ${signal.recommended_entry:>11,.2f}")
        print(f"  Stop Loss:         ${signal.stop_loss:>11,.2f}")
        print(f"  Take Profit:       ${signal.take_profit:>11,.2f}")
        print(f"  Position Size:     {signal.position_size:>11.1%}")
        
        print(f"\nCOMPONENT SCORES:")
        print(f"  Technical:    {signal.technical_score:>6.2f}")
        print(f"  Sentiment:    {signal.sentiment_score:>6.2f}")
        print(f"  On-chain:     {signal.onchain_score:>6.2f}")
        print(f"  Market:       {signal.market_score:>6.2f}")
        
        print(f"\nREASONING:")
        print(f"  {signal.reasoning}")
        
        print("\n" + "="*70)
    
    def save_results(self, signal):
        """Save results to file"""
        os.makedirs('results', exist_ok=True)
        
        filename = f"results/{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Crypto Price Predictor')
    parser.add_argument('--symbol', type=str, default='bitcoin', help='Cryptocurrency symbol')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--enhanced', action='store_true', default=True, 
                       help='Use enhanced predictor (default: True)')
    parser.add_argument('--standard', action='store_true', 
                       help='Use standard predictor instead of enhanced')
    parser.add_argument('--backtest', action='store_true', 
                       help='Run backtesting mode')
    parser.add_argument('--architecture', type=str, choices=['lstm', 'gru', 'tcn', 'hybrid'],
                       default='hybrid', help='Model architecture for enhanced predictor')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        Config.load_from_file(args.config)
    
    # Override architecture if specified
    if args.architecture:
        Config.MODEL['architecture'] = args.architecture
    
    # Determine which predictor to use
    use_enhanced = args.enhanced and not args.standard
    
    # Run system
    system = CryptoPredictionSystem(
        symbol=args.symbol, 
        train_model=args.train,
        use_enhanced=use_enhanced,
        backtest_mode=args.backtest
    )
    result = system.run()
    
    return result


if __name__ == "__main__":
    main()
