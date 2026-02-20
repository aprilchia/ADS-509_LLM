import time
import random

def sleep_politely(sleep_range=(1.0,2.0)):
    """Sleep for a random duration to respect API rate limits.

    Args:
        sleep_range: Tuple of (min, max) seconds to sleep. Defaults to (1.0, 2.0).
    """
    n = random.uniform(sleep_range[0], sleep_range[1])
    time.sleep(n)