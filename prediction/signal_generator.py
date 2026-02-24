"""
Enhanced Signal Generator

Improvements:
- Strategy confirmation: only accept BUY/SELL if ≥ min_strategy_agreement strategies agree
- Regime filter: reduce confidence in choppy/low-ADX markets
- Outputs confidence, agreement score, and regime context
"""
import numpy as np


class SignalGenerator:
    def __init__(self, confidence_threshold=0.40, min_strategy_agreement=1, 
                 adx_trending_threshold=25.0):
        """
        Args:
            confidence_threshold: Minimum model probability to generate a signal
            min_strategy_agreement: Minimum number of strategies that must agree for confirmation
            adx_trending_threshold: ADX value above which market is considered trending
        """
        self.confidence_threshold = confidence_threshold
        self.min_strategy_agreement = min_strategy_agreement
        self.adx_trending_threshold = adx_trending_threshold

    def generate_signal(self, last_prediction_probs, strategy_features=None, 
                       initial_signal=None, initial_confidence=None):
        """
        Generates a trade signal from model probabilities with strategy confirmation.
        
        Args:
            last_prediction_probs: [prob_hold, prob_buy, prob_sell]
            strategy_features: Dict with strategy context
            initial_signal: (Optional) Pre-calculated signal (e.g. from TradingSystem)
            initial_confidence: (Optional) Pre-calculated confidence
        
        Returns:
            tuple: (signal, confidence, details_dict)
        """
        hold_prob, buy_prob, sell_prob = last_prediction_probs
        
        # Default details
        details = {
            'model_confidence': max(hold_prob, buy_prob, sell_prob),
            'strategy_agreement': 0,
            'strategies_aligned': 0,
            'regime': 'unknown',
            'confidence_adjusted': False
        }
        
        # Determine raw signal from model
        if initial_signal and initial_confidence:
             raw_signal = initial_signal
             raw_confidence = float(initial_confidence)
             # If initial signal is HOLD, return immediately unless confidence is high
             if raw_signal == "HOLD":
                 return "HOLD", raw_confidence, details
        else:
            if buy_prob > self.confidence_threshold:
                raw_signal = "BUY"
                raw_confidence = float(buy_prob)
            elif sell_prob > self.confidence_threshold:
                raw_signal = "SELL"
                raw_confidence = float(sell_prob)
            else:
                return "HOLD", float(max(hold_prob, buy_prob, sell_prob)), details
        
        # ── Strategy Confirmation ──
        if strategy_features is not None:
            agreement = strategy_features.get('Strat_Agreement', 0)
            adx = strategy_features.get('Strat_ADX', 25.0)
            
            details['strategy_agreement'] = agreement
            
            # Count how many strategies align with the signal direction
            strategy_signals = [
                strategy_features.get('Strat_EMA_Cross', 0),
                strategy_features.get('Strat_Breakout', 0),
                strategy_features.get('Strat_MeanRev', 0),
                strategy_features.get('Strat_VolBreakout', 0),
                strategy_features.get('Strat_MomDiv', 0),
            ]
            
            if raw_signal == "BUY":
                aligned = sum(1 for s in strategy_signals if s > 0)
            else:  # SELL
                aligned = sum(1 for s in strategy_signals if s < 0)
            
            details['strategies_aligned'] = aligned
            
            # Strategy agreement filter
            if aligned < self.min_strategy_agreement:
                # Not enough strategies agree — downgrade to HOLD
                details['confidence_adjusted'] = True
                return "HOLD", raw_confidence * 0.5, details
            
            # Regime filter: ADX-based
            if adx >= self.adx_trending_threshold:
                details['regime'] = 'trending'
                # Boost confidence slightly in trending markets
                adjusted_confidence = min(raw_confidence * 1.1, 0.99)
            else:
                details['regime'] = 'ranging'
                # Reduce confidence in choppy markets
                adjusted_confidence = raw_confidence * 0.85
                details['confidence_adjusted'] = True
                
                # In ranging markets, only take very high-confidence signals
                if adjusted_confidence < self.confidence_threshold:
                    return "HOLD", adjusted_confidence, details
            
            return raw_signal, adjusted_confidence, details
        
        # No strategy features available — use raw signal
        return raw_signal, raw_confidence, details

    def generate_signal_simple(self, last_prediction_probs):
        """
        Simple signal generation (backward-compatible).
        Returns: (signal, confidence)
        """
        signal, confidence, _ = self.generate_signal(last_prediction_probs)
        return signal, confidence


if __name__ == "__main__":
    gen = SignalGenerator()
    
    print("=== Without Strategy Confirmation ===")
    print(gen.generate_signal([0.2, 0.7, 0.1]))
    print(gen.generate_signal([0.8, 0.1, 0.1]))
    
    print("\n=== With Strategy Confirmation (BUY, 3 strategies agree) ===")
    strat = {
        'Strat_Agreement': 3,
        'Strat_ADX': 30,
        'Strat_EMA_Cross': 1,
        'Strat_Breakout': 1,
        'Strat_MeanRev': 1,
        'Strat_VolBreakout': 0,
        'Strat_MomDiv': 0,
    }
    print(gen.generate_signal([0.15, 0.75, 0.10], strat))
    
    print("\n=== With Strategy Confirmation (BUY, only 1 strategy agrees — blocked) ===")
    strat_low = {
        'Strat_Agreement': 1,
        'Strat_ADX': 15,
        'Strat_EMA_Cross': 1,
        'Strat_Breakout': 0,
        'Strat_MeanRev': 0,
        'Strat_VolBreakout': 0,
        'Strat_MomDiv': -1,
    }
    print(gen.generate_signal([0.15, 0.72, 0.13], strat_low))
