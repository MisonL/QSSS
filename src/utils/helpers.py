import time
import functools
import logging

# Placeholder for retry_on_exception and other utility functions
# We will move the actual code here in the next steps.

def retry_on_exception(retries=3, delay=1):
    """
    A decorator to retry a function call if it raises an exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator
