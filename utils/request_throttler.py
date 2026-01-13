"""
Request throttler and rate limiter for API calls.
Implements intelligent caching and request pooling to work with API limits.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

class RequestThrottler:
    """
    Client-side rate limiter and request throttler.
    Prevents hammering the API and manages request queuing.
    """
    
    def __init__(self, min_interval: float = 2.0, max_retries: int = 5):
        """
        Initialize the throttler.
        
        Args:
            min_interval: Minimum seconds between requests
            max_retries: Maximum retry attempts
        """
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.last_request_time: Dict[str, float] = {}
        self.failed_symbols: Dict[str, Tuple[float, int]] = {}  # symbol -> (time, count)
        self.lock = threading.Lock()
        self.rate_limit_cooldown = 60  # 60 seconds minimum wait after rate limit
        
    def wait_if_needed(self, symbol: str) -> Optional[float]:
        """
        Check if we should wait before making a request.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Seconds to wait, or None if ready
        """
        with self.lock:
            # Check if symbol was recently rate limited
            if symbol in self.failed_symbols:
                fail_time, fail_count = self.failed_symbols[symbol]
                time_since_fail = time.time() - fail_time
                
                # Exponential backoff: 60s, 120s, 180s, etc.
                min_wait = self.rate_limit_cooldown * fail_count
                
                if time_since_fail < min_wait:
                    return min_wait - time_since_fail
                else:
                    # Cooldown expired, reset counter
                    del self.failed_symbols[symbol]
            
            # Check minimum interval
            last_time = self.last_request_time.get(symbol, 0)
            time_since_last = time.time() - last_time
            
            if time_since_last < self.min_interval:
                return self.min_interval - time_since_last
            
            return None
    
    def record_request(self, symbol: str):
        """Record a successful request."""
        with self.lock:
            self.last_request_time[symbol] = time.time()
    
    def record_rate_limit(self, symbol: str):
        """Record a rate limit error."""
        with self.lock:
            if symbol in self.failed_symbols:
                fail_time, fail_count = self.failed_symbols[symbol]
                self.failed_symbols[symbol] = (time.time(), fail_count + 1)
            else:
                self.failed_symbols[symbol] = (time.time(), 1)
    
    def can_retry(self, symbol: str) -> bool:
        """Check if we should retry a symbol."""
        with self.lock:
            if symbol in self.failed_symbols:
                _, fail_count = self.failed_symbols[symbol]
                return fail_count < self.max_retries
            return True
    
    def get_retry_wait_time(self, symbol: str, attempt: int) -> float:
        """Get recommended wait time before retry."""
        # Exponential backoff: 3, 6, 12, 25, 51 seconds
        base_wait = 3 * (2 ** (attempt - 1))
        # Cap at 60 seconds
        return min(base_wait, 60)
    
    def reset_symbol(self, symbol: str):
        """Reset cooldown for a symbol."""
        with self.lock:
            if symbol in self.failed_symbols:
                del self.failed_symbols[symbol]


# Global throttler instance
_throttler = RequestThrottler(min_interval=2.0, max_retries=5)


def get_throttler() -> RequestThrottler:
    """Get the global throttler instance."""
    return _throttler
