import logging
import sys

class AIMansionLogger:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIMansionLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self._logger = logging.getLogger("AIMansion")
        self._logger.setLevel(logging.INFO)
        
        # Check if handlers already exist to avoid duplicate logs
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    @classmethod
    def get_logger(cls, name: str = "AIMansion"):
        if cls._instance is None:
            cls()
        return logging.getLogger(f"AIMansion.{name}")

    @classmethod
    def set_level(cls, level: int):
        if cls._instance is None:
            cls()
        # Set level for the root AIMansion logger
        cls._instance._logger.setLevel(level)
        # Also update handlers
        for handler in cls._instance._logger.handlers:
            handler.setLevel(level)

# Convenience functions
def get_logger(name: str):
    return AIMansionLogger.get_logger(name)

def set_log_level(level: str):
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    AIMansionLogger.set_level(level_map.get(level.upper(), logging.INFO))
