from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """Abstract base for all feature extractors."""
    
    @abstractmethod
    def compute(self, data) -> dict:
        """Compute features from the input data object.
        
        Args:
            data: Any object containing the necessary information.
                  For volatility, it could be your OptionChainGreeks.
        
        Returns:
            dict: Feature names mapped to scalar values.
        """
        pass