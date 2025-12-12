"""
Alternative Data Sources Module
================================

Collects and processes alternative data signals:
- Google Trends for search interest
- NFT market activity
- Derivatives market data (funding rates, open interest)
- Social sentiment from additional sources

These alternative signals can provide early indicators
of market sentiment and price movements.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pytrends.request import TrendReq
import time

logger = logging.getLogger(__name__)


class AlternativeDataCollector:
    """
    Collects alternative data sources for enhanced prediction
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize alternative data collector
        
        Args:
            api_keys: Dictionary of API keys
        """
        self.api_keys = api_keys
        
        # Initialize Google Trends
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.trends_available = True
        except Exception as e:
            logger.warning(f"Google Trends initialization failed: {e}")
            self.trends_available = False
        
        logger.info("AlternativeDataCollector initialized")
    
    def collect_all_alternative_data(
        self,
        symbol: str,
        hours_back: int = 168
    ) -> Dict:
        """
        Collect all alternative data sources
        
        Args:
            symbol: Cryptocurrency symbol
            hours_back: Hours of historical data
            
        Returns:
            Dictionary with alternative data
        """
        logger.info(f"Collecting alternative data for {symbol}")
        
        data = {
            'google_trends': self.get_google_trends(symbol),
            'nft_activity': self.get_nft_activity(symbol),
            'derivatives': self.get_derivatives_data(symbol),
            'funding_rates': self.get_funding_rates(symbol),
            'liquidations': self.get_liquidation_data(symbol),
            'options_flow': self.get_options_flow(symbol)
        }
        
        return data
    
    def get_google_trends(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get Google Trends search interest
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with trends data
        """
        if not self.trends_available:
            logger.warning("Google Trends not available")
            return None
        
        try:
            # Map crypto symbols to search terms
            search_terms = self._get_search_terms(symbol)
            
            # Build payload
            self.pytrends.build_payload(
                search_terms,
                timeframe='today 3-m',  # Last 3 months
                geo='',  # Worldwide
                gprop=''
            )
            
            # Get interest over time
            trends_df = self.pytrends.interest_over_time()
            
            if trends_df.empty:
                logger.warning(f"No trends data for {symbol}")
                return None
            
            # Calculate aggregate score
            trends_df['aggregate_interest'] = trends_df[search_terms].mean(axis=1)
            
            # Calculate momentum
            trends_df['interest_momentum'] = trends_df['aggregate_interest'].pct_change(periods=7)
            
            # Normalize to 0-1
            trends_df['normalized_interest'] = (
                trends_df['aggregate_interest'] / trends_df['aggregate_interest'].max()
            )
            
            # Get related queries
            related_queries = self.pytrends.related_queries()
            rising_queries = self._extract_rising_queries(related_queries, search_terms)
            
            # Add metadata
            trends_df['rising_queries_count'] = len(rising_queries)
            
            logger.info(f"Collected Google Trends data: {len(trends_df)} data points")
            
            return trends_df
            
        except Exception as e:
            logger.error(f"Error collecting Google Trends: {e}")
            return None
    
    def get_nft_activity(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get NFT market activity for blockchain
        
        Args:
            symbol: Cryptocurrency symbol (e.g., ethereum)
            
        Returns:
            DataFrame with NFT metrics
        """
        try:
            # OpenSea API (requires API key)
            if 'opensea' not in self.api_keys:
                logger.warning("OpenSea API key not configured")
                return self._get_mock_nft_data()
            
            # Map symbol to blockchain
            blockchain = self._map_symbol_to_blockchain(symbol)
            
            headers = {
                'X-API-KEY': self.api_keys['opensea']
            }
            
            # Get collection stats
            url = f"https://api.opensea.io/api/v1/collections?asset_owner=&offset=0&limit=50&chain={blockchain}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"OpenSea API returned {response.status_code}")
                return self._get_mock_nft_data()
            
            data = response.json()
            
            # Process NFT data
            nft_df = pd.DataFrame([{
                'timestamp': datetime.now(),
                'total_volume': sum([c.get('stats', {}).get('total_volume', 0) for c in data.get('collections', [])]),
                'num_sales': sum([c.get('stats', {}).get('num_sales', 0) for c in data.get('collections', [])]),
                'avg_price': np.mean([c.get('stats', {}).get('average_price', 0) for c in data.get('collections', [])]),
                'active_collections': len(data.get('collections', []))
            }])
            
            # Calculate metrics
            nft_df['nft_activity_score'] = (
                (nft_df['total_volume'] / 1000) + 
                (nft_df['num_sales'] / 100)
            ) / 2
            
            logger.info(f"Collected NFT data for {blockchain}")
            
            return nft_df
            
        except Exception as e:
            logger.error(f"Error collecting NFT data: {e}")
            return self._get_mock_nft_data()
    
    def get_derivatives_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get derivatives market data
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with derivatives metrics
        """
        try:
            # Binance Futures API (public, no key needed)
            base_symbol = self._normalize_symbol(symbol).upper()
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={base_symbol}USDT"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Binance Futures API returned {response.status_code}")
                return None
            
            data = response.json()
            
            # Get long/short ratio
            ratio_url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={base_symbol}USDT&period=1h&limit=24"
            ratio_response = requests.get(ratio_url, timeout=10)
            
            derivatives_data = []
            
            if ratio_response.status_code == 200:
                ratio_data = ratio_response.json()
                
                for item in ratio_data:
                    derivatives_data.append({
                        'timestamp': pd.to_datetime(item['timestamp'], unit='ms'),
                        'open_interest': float(data.get('openInterest', 0)),
                        'long_short_ratio': float(item['longShortRatio']),
                        'long_account': float(item['longAccount']),
                        'short_account': float(item['shortAccount'])
                    })
            
            derivatives_df = pd.DataFrame(derivatives_data)
            
            if not derivatives_df.empty:
                # Calculate metrics
                derivatives_df['oi_momentum'] = derivatives_df['open_interest'].pct_change()
                derivatives_df['sentiment_score'] = (
                    derivatives_df['long_short_ratio'] - 1
                ) / (derivatives_df['long_short_ratio'] + 1)  # Normalize to [-1, 1]
                
                # Extreme positioning indicator
                derivatives_df['extreme_positioning'] = (
                    (derivatives_df['long_short_ratio'] > 2.5) | 
                    (derivatives_df['long_short_ratio'] < 0.4)
                ).astype(int)
                
                logger.info(f"Collected derivatives data: {len(derivatives_df)} points")
            
            return derivatives_df
            
        except Exception as e:
            logger.error(f"Error collecting derivatives data: {e}")
            return None
    
    def get_funding_rates(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get perpetual futures funding rates
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with funding rate data
        """
        try:
            base_symbol = self._normalize_symbol(symbol).upper()
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={base_symbol}USDT&limit=100"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Funding rate API returned {response.status_code}")
                return None
            
            data = response.json()
            
            funding_df = pd.DataFrame(data)
            funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
            funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)
            
            # Calculate metrics
            funding_df['funding_rate_ma'] = funding_df['fundingRate'].rolling(8).mean()
            funding_df['funding_momentum'] = funding_df['fundingRate'].diff()
            
            # Annualized funding rate (3 times per day)
            funding_df['annualized_funding'] = funding_df['fundingRate'] * 365 * 3
            
            # Extreme funding indicator
            funding_df['extreme_funding'] = (
                (funding_df['fundingRate'] > 0.0005) |  # Very positive (longs paying)
                (funding_df['fundingRate'] < -0.0005)   # Very negative (shorts paying)
            ).astype(int)
            
            logger.info(f"Collected funding rates: {len(funding_df)} points")
            
            return funding_df
            
        except Exception as e:
            logger.error(f"Error collecting funding rates: {e}")
            return None
    
    def get_liquidation_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get liquidation data (approximation via price movements)
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with liquidation estimates
        """
        try:
            # Note: Real liquidation data requires paid APIs (e.g., Coinglass)
            # This is a simplified version
            
            base_symbol = self._normalize_symbol(symbol).upper()
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={base_symbol}USDT&interval=1h&limit=168"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            liquidation_data = []
            for candle in data:
                timestamp = pd.to_datetime(candle[0], unit='ms')
                open_price = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5])
                
                # Estimate liquidations based on sharp price movements
                price_range = (high - low) / open_price
                
                # Potential long liquidations (sharp drop)
                long_liquidation_estimate = max(0, (open_price - low) / open_price - 0.02) * volume
                
                # Potential short liquidations (sharp rise)
                short_liquidation_estimate = max(0, (high - open_price) / open_price - 0.02) * volume
                
                liquidation_data.append({
                    'timestamp': timestamp,
                    'price_range': price_range,
                    'estimated_long_liq': long_liquidation_estimate,
                    'estimated_short_liq': short_liquidation_estimate,
                    'total_estimated_liq': long_liquidation_estimate + short_liquidation_estimate
                })
            
            liquidation_df = pd.DataFrame(liquidation_data)
            
            # Calculate liquidation cascades (clusters of high liquidations)
            liquidation_df['liq_cascade'] = (
                liquidation_df['total_estimated_liq'] > 
                liquidation_df['total_estimated_liq'].rolling(10).mean() * 2
            ).astype(int)
            
            logger.info(f"Estimated liquidation data: {len(liquidation_df)} points")
            
            return liquidation_df
            
        except Exception as e:
            logger.error(f"Error collecting liquidation data: {e}")
            return None
    
    def get_options_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get options flow data (if available)
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with options metrics
        """
        try:
            # Deribit is main crypto options exchange
            base_symbol = self._normalize_symbol(symbol).upper()
            
            url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={base_symbol}&kind=option"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Deribit API returned {response.status_code}")
                return None
            
            data = response.json()
            
            if 'result' not in data or not data['result']:
                return None
            
            options_data = []
            for option in data['result']:
                options_data.append({
                    'instrument_name': option['instrument_name'],
                    'volume': option.get('volume', 0),
                    'open_interest': option.get('open_interest', 0),
                    'bid_price': option.get('bid_price', 0),
                    'ask_price': option.get('ask_price', 0),
                    'mark_price': option.get('mark_price', 0)
                })
            
            options_df = pd.DataFrame(options_data)
            
            if not options_df.empty:
                # Aggregate metrics
                total_volume = options_df['volume'].sum()
                total_oi = options_df['open_interest'].sum()
                
                summary = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'total_options_volume': total_volume,
                    'total_open_interest': total_oi,
                    'put_call_ratio': self._calculate_put_call_ratio(options_df),
                    'options_activity_score': np.log1p(total_volume + total_oi)
                }])
                
                logger.info(f"Collected options flow data")
                
                return summary
            
            return None
            
        except Exception as e:
            logger.error(f"Error collecting options flow: {e}")
            return None
    
    def _get_search_terms(self, symbol: str) -> List[str]:
        """Get relevant search terms for crypto"""
        symbol_lower = symbol.lower()
        
        search_map = {
            'bitcoin': ['Bitcoin', 'BTC', 'Bitcoin price'],
            'ethereum': ['Ethereum', 'ETH', 'Ethereum price'],
            'cardano': ['Cardano', 'ADA'],
            'solana': ['Solana', 'SOL'],
            'ripple': ['Ripple', 'XRP'],
            'dogecoin': ['Dogecoin', 'DOGE']
        }
        
        return search_map.get(symbol_lower, [symbol.capitalize(), symbol.upper()])[:3]
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol"""
        symbol_map = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'cardano': 'ADA',
            'solana': 'SOL',
            'ripple': 'XRP',
            'dogecoin': 'DOGE'
        }
        return symbol_map.get(symbol.lower(), symbol.upper())
    
    def _map_symbol_to_blockchain(self, symbol: str) -> str:
        """Map crypto symbol to blockchain"""
        blockchain_map = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'solana': 'solana',
            'cardano': 'cardano'
        }
        return blockchain_map.get(symbol.lower(), 'ethereum')
    
    def _get_mock_nft_data(self) -> pd.DataFrame:
        """Generate mock NFT data"""
        return pd.DataFrame([{
            'timestamp': datetime.now(),
            'total_volume': 0,
            'num_sales': 0,
            'avg_price': 0,
            'active_collections': 0,
            'nft_activity_score': 0
        }])
    
    def _extract_rising_queries(self, related_queries: Dict, search_terms: List[str]) -> List[str]:
        """Extract rising search queries"""
        rising = []
        
        for term in search_terms:
            if term in related_queries and 'rising' in related_queries[term]:
                rising_df = related_queries[term]['rising']
                if rising_df is not None and not rising_df.empty:
                    rising.extend(rising_df['query'].tolist()[:5])
        
        return rising
    
    def _calculate_put_call_ratio(self, options_df: pd.DataFrame) -> float:
        """Calculate put/call ratio from options data"""
        try:
            puts = options_df[options_df['instrument_name'].str.contains('-P')]
            calls = options_df[options_df['instrument_name'].str.contains('-C')]
            
            put_volume = puts['volume'].sum()
            call_volume = calls['volume'].sum()
            
            if call_volume > 0:
                return put_volume / call_volume
            else:
                return 1.0
        except:
            return 1.0
    
    def get_alternative_data_features(self, data: Dict) -> Dict[str, float]:
        """
        Extract features from alternative data
        
        Args:
            data: Dictionary with alternative data
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Google Trends features
        if data.get('google_trends') is not None and not data['google_trends'].empty:
            trends = data['google_trends']
            features['trends_interest'] = trends['normalized_interest'].iloc[-1]
            features['trends_momentum'] = trends['interest_momentum'].iloc[-1]
            features['trends_rising_queries'] = trends['rising_queries_count'].iloc[-1]
        else:
            features['trends_interest'] = 0.5
            features['trends_momentum'] = 0.0
            features['trends_rising_queries'] = 0
        
        # NFT features
        if data.get('nft_activity') is not None and not data['nft_activity'].empty:
            nft = data['nft_activity']
            features['nft_activity'] = nft['nft_activity_score'].iloc[-1]
        else:
            features['nft_activity'] = 0.0
        
        # Derivatives features
        if data.get('derivatives') is not None and not data['derivatives'].empty:
            deriv = data['derivatives']
            features['long_short_ratio'] = deriv['long_short_ratio'].iloc[-1]
            features['sentiment_from_positioning'] = deriv['sentiment_score'].iloc[-1]
            features['extreme_positioning'] = deriv['extreme_positioning'].iloc[-1]
        else:
            features['long_short_ratio'] = 1.0
            features['sentiment_from_positioning'] = 0.0
            features['extreme_positioning'] = 0
        
        # Funding rate features
        if data.get('funding_rates') is not None and not data['funding_rates'].empty:
            funding = data['funding_rates']
            features['funding_rate'] = funding['fundingRate'].iloc[-1]
            features['annualized_funding'] = funding['annualized_funding'].iloc[-1]
            features['extreme_funding'] = funding['extreme_funding'].iloc[-1]
        else:
            features['funding_rate'] = 0.0
            features['annualized_funding'] = 0.0
            features['extreme_funding'] = 0
        
        # Liquidation features
        if data.get('liquidations') is not None and not data['liquidations'].empty:
            liq = data['liquidations']
            features['liquidation_risk'] = liq['total_estimated_liq'].iloc[-1]
            features['liquidation_cascade'] = liq['liq_cascade'].iloc[-1]
        else:
            features['liquidation_risk'] = 0.0
            features['liquidation_cascade'] = 0
        
        # Options features
        if data.get('options_flow') is not None and not data['options_flow'].empty:
            opts = data['options_flow']
            features['put_call_ratio'] = opts['put_call_ratio'].iloc[-1]
            features['options_activity'] = opts['options_activity_score'].iloc[-1]
        else:
            features['put_call_ratio'] = 1.0
            features['options_activity'] = 0.0
        
        return features
