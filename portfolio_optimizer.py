"""
Portfolio Optimization Module
==============================

Multi-asset portfolio optimization and position sizing:
- Modern Portfolio Theory (MPT)
- Kelly Criterion for optimal position sizing
- Risk parity allocation
- Multi-asset signal coordination
- Portfolio rebalancing

Generates portfolio-aware trading signals considering
correlations and diversification across multiple assets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssetSignal:
    """Individual asset trading signal"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    predicted_return: float
    predicted_volatility: float
    current_price: float
    timestamp: pd.Timestamp


@dataclass
class PortfolioPosition:
    """Current portfolio position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    weight: float
    unrealized_pnl: float


class PortfolioOptimizer:
    """
    Portfolio optimization for multi-asset crypto trading
    """
    
    def __init__(
        self,
        total_capital: float = 10000.0,
        max_position_size: float = 0.2,
        risk_free_rate: float = 0.02,
        rebalance_threshold: float = 0.05
    ):
        """
        Initialize portfolio optimizer
        
        Args:
            total_capital: Total capital available
            max_position_size: Maximum position size per asset (20%)
            risk_free_rate: Risk-free rate for Sharpe ratio
            rebalance_threshold: Threshold for triggering rebalance (5%)
        """
        self.total_capital = total_capital
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.rebalance_threshold = rebalance_threshold
        
        # Current portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash = total_capital
        
        # Historical data for optimization
        self.returns_history: Dict[str, List[float]] = {}
        self.correlation_matrix = None
        
        logger.info("PortfolioOptimizer initialized")
    
    def optimize_portfolio(
        self,
        signals: List[AssetSignal],
        historical_returns: Optional[pd.DataFrame] = None,
        optimization_method: str = 'sharpe'
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation across multiple assets
        
        Args:
            signals: List of signals for different assets
            historical_returns: DataFrame with historical returns
            optimization_method: 'sharpe', 'min_variance', 'risk_parity', 'kelly'
            
        Returns:
            Dictionary of optimal weights per asset
        """
        logger.info(f"Optimizing portfolio with {len(signals)} signals")
        
        # Filter to actionable signals
        actionable_signals = [s for s in signals if s.signal_type in ['BUY', 'SELL'] 
                             and s.confidence > 0.65]
        
        if len(actionable_signals) == 0:
            logger.info("No actionable signals, maintaining current positions")
            return self._get_current_weights()
        
        # Build expected returns and covariance
        expected_returns, cov_matrix = self._build_optimization_inputs(
            actionable_signals, historical_returns
        )
        
        # Optimize based on method
        if optimization_method == 'sharpe':
            weights = self._maximize_sharpe_ratio(expected_returns, cov_matrix)
        elif optimization_method == 'min_variance':
            weights = self._minimize_variance(cov_matrix)
        elif optimization_method == 'risk_parity':
            weights = self._risk_parity(cov_matrix)
        elif optimization_method == 'kelly':
            weights = self._kelly_criterion(expected_returns, cov_matrix)
        else:
            weights = self._maximize_sharpe_ratio(expected_returns, cov_matrix)
        
        # Map weights back to symbols
        symbol_weights = {
            sig.symbol: weights[i] 
            for i, sig in enumerate(actionable_signals)
        }
        
        # Apply constraints
        symbol_weights = self._apply_constraints(symbol_weights, signals)
        
        logger.info(f"Optimal portfolio weights: {symbol_weights}")
        
        return symbol_weights
    
    def calculate_position_sizes(
        self,
        optimal_weights: Dict[str, float],
        signals: List[AssetSignal]
    ) -> Dict[str, Dict]:
        """
        Calculate specific position sizes for each asset
        
        Args:
            optimal_weights: Optimal portfolio weights
            signals: List of asset signals
            
        Returns:
            Dictionary with position details per asset
        """
        positions = {}
        
        # Calculate available capital
        available_capital = self.cash + sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        
        signal_map = {s.symbol: s for s in signals}
        
        for symbol, weight in optimal_weights.items():
            if weight == 0 or symbol not in signal_map:
                continue
            
            signal = signal_map[symbol]
            
            # Calculate dollar allocation
            dollar_allocation = available_capital * weight
            
            # Calculate quantity
            quantity = dollar_allocation / signal.current_price
            
            # Adjust for existing position
            current_position = self.positions.get(symbol)
            if current_position:
                quantity_change = quantity - current_position.quantity
                action = 'INCREASE' if quantity_change > 0 else 'DECREASE'
            else:
                quantity_change = quantity
                action = 'OPEN'
            
            positions[symbol] = {
                'target_weight': weight,
                'dollar_allocation': dollar_allocation,
                'target_quantity': quantity,
                'quantity_change': quantity_change,
                'action': action,
                'signal_confidence': signal.confidence,
                'expected_return': signal.predicted_return,
                'risk': signal.predicted_volatility
            }
        
        return positions
    
    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """
        Determine if portfolio needs rebalancing
        
        Args:
            current_weights: Current portfolio weights
            
        Returns:
            True if rebalancing needed
        """
        target_weights = self._get_target_weights()
        
        # Calculate maximum deviation
        max_deviation = 0
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            deviation = abs(current - target)
            max_deviation = max(max_deviation, deviation)
        
        needs_rebalance = max_deviation > self.rebalance_threshold
        
        if needs_rebalance:
            logger.info(f"Rebalancing needed (max deviation: {max_deviation:.2%})")
        
        return needs_rebalance
    
    def calculate_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics
        
        Returns:
            Dictionary with portfolio metrics
        """
        total_value = self.cash
        total_pnl = 0
        
        for pos in self.positions.values():
            position_value = pos.quantity * pos.current_price
            total_value += position_value
            total_pnl += pos.unrealized_pnl
        
        # Calculate returns
        total_return = (total_value - self.total_capital) / self.total_capital
        
        # Calculate volatility (if we have history)
        portfolio_volatility = self._calculate_portfolio_volatility()
        
        # Calculate Sharpe ratio
        if portfolio_volatility > 0:
            sharpe_ratio = (total_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0
        
        # Concentration (Herfindahl index)
        weights = [pos.weight for pos in self.positions.values()]
        concentration = sum(w**2 for w in weights) if weights else 0
        
        metrics = {
            'total_value': total_value,
            'cash': self.cash,
            'invested': total_value - self.cash,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'num_positions': len(self.positions),
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'concentration': concentration,
            'diversification': 1 / concentration if concentration > 0 else 0
        }
        
        return metrics
    
    def _maximize_sharpe_ratio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Maximize Sharpe ratio"""
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def _minimize_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimize portfolio variance"""
        n_assets = cov_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def _risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity allocation - equal risk contribution"""
        n_assets = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def _kelly_criterion(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Kelly criterion for optimal bet sizing"""
        # Simplified Kelly: f = (p*b - q) / b
        # For continuous case: f = μ / σ²
        
        n_assets = len(expected_returns)
        
        # Calculate Kelly fractions
        kelly_fractions = []
        for i in range(n_assets):
            if cov_matrix[i, i] > 0:
                kelly_f = expected_returns[i] / cov_matrix[i, i]
                # Apply half-Kelly for safety
                kelly_f = kelly_f * 0.5
                # Clip to reasonable bounds
                kelly_f = np.clip(kelly_f, 0, self.max_position_size)
                kelly_fractions.append(kelly_f)
            else:
                kelly_fractions.append(0)
        
        # Normalize to sum to 1
        kelly_fractions = np.array(kelly_fractions)
        total = np.sum(kelly_fractions)
        
        if total > 0:
            weights = kelly_fractions / total
        else:
            weights = np.array([1/n_assets] * n_assets)
        
        return weights
    
    def _build_optimization_inputs(
        self,
        signals: List[AssetSignal],
        historical_returns: Optional[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build expected returns and covariance matrix"""
        n_assets = len(signals)
        
        # Expected returns from signals
        expected_returns = np.array([s.predicted_return for s in signals])
        
        # Build covariance matrix
        if historical_returns is not None and not historical_returns.empty:
            # Use historical covariance
            symbols = [s.symbol for s in signals]
            available_symbols = [s for s in symbols if s in historical_returns.columns]
            
            if len(available_symbols) >= 2:
                cov_matrix = historical_returns[available_symbols].cov().values
            else:
                # Use predicted volatilities
                volatilities = np.array([s.predicted_volatility for s in signals])
                cov_matrix = np.diag(volatilities ** 2)
        else:
            # Use predicted volatilities
            volatilities = np.array([s.predicted_volatility for s in signals])
            cov_matrix = np.diag(volatilities ** 2)
            
            # Assume some correlation (conservative estimate)
            correlation = 0.3
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    cov_matrix[i, j] = correlation * volatilities[i] * volatilities[j]
                    cov_matrix[j, i] = cov_matrix[i, j]
        
        return expected_returns, cov_matrix
    
    def _apply_constraints(
        self,
        weights: Dict[str, float],
        all_signals: List[AssetSignal]
    ) -> Dict[str, float]:
        """Apply portfolio constraints"""
        # Filter out low-confidence signals
        signal_map = {s.symbol: s for s in all_signals}
        
        adjusted_weights = {}
        for symbol, weight in weights.items():
            if symbol in signal_map:
                signal = signal_map[symbol]
                
                # Scale weight by confidence
                confidence_adjusted = weight * (signal.confidence ** 0.5)
                
                # Apply maximum position constraint
                final_weight = min(confidence_adjusted, self.max_position_size)
                
                if final_weight > 0.01:  # Minimum 1% position
                    adjusted_weights[symbol] = final_weight
        
        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        total_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        
        if total_value == 0:
            return {}
        
        return {
            symbol: (pos.quantity * pos.current_price) / total_value
            for symbol, pos in self.positions.items()
        }
    
    def _get_target_weights(self) -> Dict[str, float]:
        """Get target weights (placeholder)"""
        # This would be set by the optimization
        # For now, return current weights
        return self._get_current_weights()
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if not self.returns_history:
            return 0.0
        
        # Get portfolio returns history
        portfolio_returns = self._calculate_portfolio_returns_history()
        
        if len(portfolio_returns) < 2:
            return 0.0
        
        return np.std(portfolio_returns) * np.sqrt(365)  # Annualized
    
    def _calculate_portfolio_returns_history(self) -> List[float]:
        """Calculate historical portfolio returns"""
        # Placeholder - would need actual historical tracking
        return []
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        action: str
    ):
        """
        Update a portfolio position
        
        Args:
            symbol: Asset symbol
            quantity: Quantity to trade
            price: Execution price
            action: 'BUY', 'SELL', 'CLOSE'
        """
        if action == 'BUY':
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                total_cost = (pos.quantity * pos.entry_price) + (quantity * price)
                total_quantity = pos.quantity + quantity
                new_entry_price = total_cost / total_quantity
                
                pos.quantity = total_quantity
                pos.entry_price = new_entry_price
            else:
                # Open new position
                self.positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    weight=0.0,
                    unrealized_pnl=0.0
                )
            
            self.cash -= quantity * price
            
        elif action == 'SELL':
            if symbol in self.positions:
                pos = self.positions[symbol]
                sell_quantity = min(quantity, pos.quantity)
                
                # Realize P&L
                pnl = sell_quantity * (price - pos.entry_price)
                
                pos.quantity -= sell_quantity
                self.cash += sell_quantity * price
                
                # Close position if fully sold
                if pos.quantity <= 0:
                    del self.positions[symbol]
                
                logger.info(f"Sold {sell_quantity} {symbol}, realized P&L: ${pnl:.2f}")
        
        # Update weights
        self._update_weights()
    
    def _update_weights(self):
        """Update position weights"""
        total_value = self.cash + sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        
        for pos in self.positions.values():
            position_value = pos.quantity * pos.current_price
            pos.weight = position_value / total_value if total_value > 0 else 0
            pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
