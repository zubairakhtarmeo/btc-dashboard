"""
Advanced Features Integration
==============================

Integrates regime detection, alternative data, portfolio optimization,
and explainability into the main prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from regime_detector import RegimeDetector
from alternative_data import AlternativeDataCollector
from portfolio_optimizer import PortfolioOptimizer, AssetSignal
from explainability import ModelExplainer

logger = logging.getLogger(__name__)


class AdvancedFeaturesManager:
    """
    Manages advanced features integration
    """
    
    def __init__(self, config):
        """
        Initialize advanced features
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.regime_detector = None
        if config.REGIME_DETECTION['enabled']:
            self.regime_detector = RegimeDetector(
                lookback_window=config.REGIME_DETECTION['lookback_window'],
                volatility_window=config.REGIME_DETECTION['volatility_window'],
                regime_threshold=config.REGIME_DETECTION['regime_threshold']
            )
            logger.info("Regime detection enabled")
        
        self.alternative_data = None
        if config.ALTERNATIVE_DATA['enabled']:
            self.alternative_data = AlternativeDataCollector(config.API_KEYS)
            logger.info("Alternative data collection enabled")
        
        self.portfolio_optimizer = None
        if config.PORTFOLIO['enabled']:
            self.portfolio_optimizer = PortfolioOptimizer(
                total_capital=config.PORTFOLIO['total_capital'],
                max_position_size=config.PORTFOLIO['max_position_size'],
                risk_free_rate=config.PORTFOLIO['risk_free_rate'],
                rebalance_threshold=config.PORTFOLIO['rebalance_threshold']
            )
            logger.info("Portfolio optimization enabled")
        
        self.explainer = None
        # Explainer initialized after model is created
        
        logger.info("AdvancedFeaturesManager initialized")
    
    def detect_regime(self, price_df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect market regime and get adaptive parameters
        
        Args:
            price_df: Price data DataFrame
            
        Returns:
            Regime information dictionary
        """
        if self.regime_detector is None:
            return None
        
        regime_info = self.regime_detector.detect_regime(price_df)
        
        logger.info(
            f"Market regime: {regime_info['primary_regime'].value} "
            f"(confidence: {regime_info['confidence']:.1%})"
        )
        
        return regime_info
    
    def collect_alternative_data(
        self,
        symbol: str,
        hours_back: int = 168
    ) -> Optional[Dict]:
        """
        Collect alternative data sources
        
        Args:
            symbol: Cryptocurrency symbol
            hours_back: Hours of historical data
            
        Returns:
            Alternative data dictionary
        """
        if self.alternative_data is None:
            return None
        
        alt_data = self.alternative_data.collect_all_alternative_data(
            symbol, hours_back
        )
        
        # Extract features
        alt_features = self.alternative_data.get_alternative_data_features(alt_data)
        
        logger.info(f"Collected {len(alt_features)} alternative data features")
        
        return {
            'raw_data': alt_data,
            'features': alt_features
        }
    
    def enhance_features_with_regime(
        self,
        features_df: pd.DataFrame,
        regime_info: Dict
    ) -> pd.DataFrame:
        """
        Add regime-based features to dataset
        
        Args:
            features_df: Original features DataFrame
            regime_info: Regime detection results
            
        Returns:
            Enhanced features DataFrame
        """
        if regime_info is None:
            return features_df
        
        # Add regime indicator features
        from regime_detector import MarketRegime, VolatilityRegime
        
        # One-hot encode regime
        regime_encodings = {
            MarketRegime.BULL_TRENDING: [1, 0, 0, 0, 0, 0],
            MarketRegime.BEAR_TRENDING: [0, 1, 0, 0, 0, 0],
            MarketRegime.SIDEWAYS: [0, 0, 1, 0, 0, 0],
            MarketRegime.HIGH_VOLATILITY: [0, 0, 0, 1, 0, 0],
            MarketRegime.LOW_VOLATILITY: [0, 0, 0, 0, 1, 0],
            MarketRegime.MEAN_REVERTING: [0, 0, 0, 0, 0, 1]
        }
        
        regime_vector = regime_encodings.get(
            regime_info['primary_regime'],
            [0, 0, 0, 0, 0, 0]
        )
        
        regime_columns = [
            'regime_bull', 'regime_bear', 'regime_sideways',
            'regime_high_vol', 'regime_low_vol', 'regime_mean_reverting'
        ]
        
        for col, val in zip(regime_columns, regime_vector):
            features_df[col] = val
        
        # Add continuous regime features
        features_df['regime_confidence'] = regime_info['confidence']
        features_df['regime_duration'] = regime_info['duration']
        features_df['volatility_percentile'] = regime_info['volatility_percentile']
        
        # Trend features
        features_df['trend_strength'] = regime_info['trend']['strength']
        features_df['trend_slope'] = regime_info['trend']['slope']
        
        # Volatility features
        features_df['volatility_current'] = regime_info['volatility']['current']
        features_df['volatility_expanding'] = int(regime_info['volatility']['expanding'])
        
        logger.info(f"Added {len(regime_columns) + 7} regime-based features")
        
        return features_df
    
    def enhance_features_with_alternative_data(
        self,
        features_df: pd.DataFrame,
        alt_data: Dict
    ) -> pd.DataFrame:
        """
        Add alternative data features
        
        Args:
            features_df: Original features DataFrame
            alt_data: Alternative data dictionary
            
        Returns:
            Enhanced features DataFrame
        """
        if alt_data is None or 'features' not in alt_data:
            return features_df
        
        # Add alternative data features
        for key, value in alt_data['features'].items():
            # Broadcast scalar value to all rows
            features_df[f'alt_{key}'] = value
        
        logger.info(f"Added {len(alt_data['features'])} alternative data features")
        
        return features_df
    
    def adapt_model_to_regime(
        self,
        predictor,
        regime_info: Dict
    ):
        """
        Adapt model parameters based on regime
        
        Args:
            predictor: Model predictor object
            regime_info: Regime information
        """
        if regime_info is None or not self.config.REGIME_DETECTION['adapt_model_parameters']:
            return
        
        adaptive_params = regime_info['adaptive_params']
        
        logger.info("Adapting model parameters to regime")
        logger.info(f"  Stop loss multiplier: {adaptive_params['stop_loss_multiplier']:.2f}")
        logger.info(f"  Position size multiplier: {adaptive_params['position_size_multiplier']:.2f}")
        logger.info(f"  Preferred horizon: {adaptive_params['preferred_horizon']}h")
        
        # Store adaptive parameters for signal generation
        if not hasattr(predictor, 'adaptive_params'):
            predictor.adaptive_params = {}
        predictor.adaptive_params.update(adaptive_params)
    
    def optimize_portfolio(
        self,
        signals: List[Dict],
        historical_returns: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Optimize portfolio across multiple assets
        
        Args:
            signals: List of trading signals for different assets
            historical_returns: Historical returns for covariance estimation
            
        Returns:
            Portfolio optimization results
        """
        if self.portfolio_optimizer is None or len(signals) < 2:
            return None
        
        # Convert signals to AssetSignal objects
        asset_signals = []
        for sig in signals:
            asset_signals.append(AssetSignal(
                symbol=sig.get('symbol', 'unknown'),
                signal_type=sig.get('signal_type', 'HOLD'),
                confidence=sig.get('confidence', 0.5),
                predicted_return=sig.get('expected_return', 0.0) / 100,  # Convert to decimal
                predicted_volatility=sig.get('volatility', 0.02),
                current_price=sig.get('current_price', 0.0),
                timestamp=pd.Timestamp.now()
            ))
        
        # Optimize portfolio
        optimal_weights = self.portfolio_optimizer.optimize_portfolio(
            asset_signals,
            historical_returns,
            optimization_method=self.config.PORTFOLIO['optimization_method']
        )
        
        # Calculate position sizes
        position_sizes = self.portfolio_optimizer.calculate_position_sizes(
            optimal_weights,
            asset_signals
        )
        
        # Get portfolio metrics
        portfolio_metrics = self.portfolio_optimizer.calculate_portfolio_metrics()
        
        logger.info("Portfolio optimization complete")
        logger.info(f"  Optimal weights: {optimal_weights}")
        logger.info(f"  Portfolio Sharpe: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
        
        return {
            'weights': optimal_weights,
            'positions': position_sizes,
            'metrics': portfolio_metrics
        }
    
    def initialize_explainer(self, model, feature_names: List[str]):
        """
        Initialize model explainer
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        if self.config.EXPLAINABILITY['enabled']:
            self.explainer = ModelExplainer(model, feature_names)
            logger.info("Model explainer initialized")
    
    def explain_prediction(
        self,
        X: np.ndarray,
        prediction: Dict,
        save_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generate explanation for prediction
        
        Args:
            X: Input features
            prediction: Model prediction
            save_path: Path to save visualizations
            
        Returns:
            Explanation dictionary
        """
        if self.explainer is None:
            return None
        
        explanation = self.explainer.explain_prediction(X, prediction, method='all')
        
        # Generate visualizations
        if self.config.EXPLAINABILITY['generate_visualizations'] and save_path:
            self.explainer.visualize_explanation(explanation, save_path)
        
        # Generate report
        if self.config.EXPLAINABILITY['save_explanations']:
            import os
            os.makedirs(self.config.EXPLAINABILITY['explanation_path'], exist_ok=True)
            report_path = os.path.join(
                self.config.EXPLAINABILITY['explanation_path'],
                f"explanation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            self.explainer.create_explanation_report(explanation, report_path)
        
        logger.info("Prediction explained")
        logger.info(f"  {explanation.get('narrative', 'No narrative available')}")
        
        return explanation
    
    def get_regime_summary(self) -> Optional[Dict]:
        """Get current regime summary"""
        if self.regime_detector is None:
            return None
        return self.regime_detector.get_regime_summary()
