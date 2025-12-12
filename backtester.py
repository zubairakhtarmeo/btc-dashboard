"""
Backtesting Framework
====================

Walk-forward validation and rolling window backtesting
for realistic performance evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """
    Walk-forward backtesting with rolling window
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        transaction_cost: float = 0.001  # 0.1%
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
        self.trades = []
        self.equity_curve = []
        
        logger.info(f"Backtester initialized (capital: ${initial_capital:,.2f})")
    
    def run_backtest(
        self,
        features_df: pd.DataFrame,
        predictor,
        signal_generator,
        test_start_idx: int = None,
        walk_forward_window: int = 720  # 30 days
    ) -> Dict:
        """
        Run walk-forward backtest
        
        Args:
            features_df: DataFrame with features
            predictor: Trained model
            signal_generator: Signal generator
            test_start_idx: Where to start testing
            walk_forward_window: Training window size
            
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting walk-forward backtest...")
        
        if test_start_idx is None:
            test_start_idx = walk_forward_window + predictor.sequence_length
        
        portfolio_value = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        
        for i in range(test_start_idx, len(features_df)):
            current_time = features_df.index[i]
            current_price = features_df['close'].iloc[i]
            
            # Get features for prediction
            test_features = features_df.iloc[i-predictor.sequence_length:i]
            
            try:
                # Prepare sequence
                X, _ = predictor.prepare_sequences_multihorizon(
                    test_features.assign(close=test_features['close'])
                )
                
                if len(X) == 0:
                    continue
                
                # Predict
                predictions = predictor.predict_with_uncertainty(X[-1:])
                
                # Generate signal
                ml_pred = {
                    'price': predictor.scaler_y.inverse_transform(
                        predictions['price_24h'][-1].reshape(-1, 1)
                    )[0, 0],
                    'direction': np.argmax(predictions['direction_24h'][-1]),
                    'direction_confidence': np.max(predictions['direction_24h'][-1]),
                    'volatility': predictions['volatility_24h'][-1][0]
                }
                
                signal = signal_generator.generate_signal(
                    ml_pred,
                    features_df.iloc[[i]],
                    current_price
                )
                
                # Execute trades based on signal
                if signal.signal_type.value in ['BUY', 'STRONG_BUY'] and position <= 0:
                    # Enter long position
                    if position == -1:
                        # Close short first
                        pnl = (entry_price - current_price) / entry_price
                        portfolio_value *= (1 + pnl - self.transaction_cost)
                    
                    # Open long
                    position = 1
                    entry_price = current_price * (1 + self.transaction_cost)
                    
                    self.trades.append({
                        'timestamp': current_time,
                        'action': 'BUY',
                        'price': entry_price,
                        'confidence': signal.confidence,
                        'signal_strength': signal.strength
                    })
                
                elif signal.signal_type.value in ['SELL', 'STRONG_SELL'] and position >= 0:
                    # Enter short or close long
                    if position == 1:
                        # Close long
                        pnl = (current_price - entry_price) / entry_price
                        portfolio_value *= (1 + pnl - self.transaction_cost)
                    
                    # Open short (if allowed)
                    position = -1
                    entry_price = current_price * (1 - self.transaction_cost)
                    
                    self.trades.append({
                        'timestamp': current_time,
                        'action': 'SELL',
                        'price': entry_price,
                        'confidence': signal.confidence,
                        'signal_strength': signal.strength
                    })
                
                elif signal.signal_type.value == 'HOLD' and position != 0:
                    # Potentially close position if confidence is low
                    if signal.confidence < 0.5:
                        if position == 1:
                            pnl = (current_price - entry_price) / entry_price
                            portfolio_value *= (1 + pnl - self.transaction_cost)
                        elif position == -1:
                            pnl = (entry_price - current_price) / entry_price
                            portfolio_value *= (1 + pnl - self.transaction_cost)
                        
                        position = 0
                
            except Exception as e:
                logger.warning(f"Error at step {i}: {e}")
                continue
            
            # Track equity
            if position == 1:
                unrealized_pnl = (current_price - entry_price) / entry_price
                current_equity = portfolio_value * (1 + unrealized_pnl)
            elif position == -1:
                unrealized_pnl = (entry_price - current_price) / entry_price
                current_equity = portfolio_value * (1 + unrealized_pnl)
            else:
                current_equity = portfolio_value
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'position': position
            })
        
        # Close any open position
        if position != 0:
            final_price = features_df['close'].iloc[-1]
            if position == 1:
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price
            portfolio_value *= (1 + pnl - self.transaction_cost)
        
        # Calculate metrics
        results = self.calculate_metrics(portfolio_value, features_df)
        
        logger.info("Backtest complete!")
        logger.info(f"  Final Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"  Total Return: {results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {results['win_rate']:.2%}")
        logger.info(f"  Total Trades: {results['total_trades']}")
        
        return results
    
    def calculate_metrics(self, final_value: float, features_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        # Total return
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Returns series
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Sharpe ratio (annualized)
        if len(equity_df) > 1:
            sharpe = (
                equity_df['returns'].mean() / (equity_df['returns'].std() + 1e-8) * 
                np.sqrt(365 * 24)  # Hourly data
            )
        else:
            sharpe = 0
        
        # Max drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        if len(trades_df) > 1:
            trades_df['pnl'] = trades_df['price'].diff()
            trades_df['is_win'] = trades_df['pnl'] > 0
            win_rate = trades_df['is_win'].sum() / len(trades_df)
        else:
            win_rate = 0
        
        # Buy & Hold comparison
        buy_hold_return = (
            features_df['close'].iloc[-1] - features_df['close'].iloc[0]
        ) / features_df['close'].iloc[0]
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades_df),
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'equity_curve': equity_df,
            'trades': trades_df
        }
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Equity curve
            equity_df = results['equity_curve']
            axes[0].plot(equity_df['timestamp'], equity_df['equity'], label='Strategy')
            axes[0].set_title('Equity Curve')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].legend()
            axes[0].grid(True)
            
            # Drawdown
            axes[1].fill_between(
                equity_df['timestamp'],
                equity_df['drawdown'],
                0,
                color='red',
                alpha=0.3
            )
            axes[1].set_title('Drawdown')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].set_xlabel('Time')
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300)
            logger.info("Results plot saved to backtest_results.png")
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")


if __name__ == "__main__":
    """Test backtester"""
    print("Backtesting framework ready!")
    print("Usage: Initialize backtester and run on historical data")
