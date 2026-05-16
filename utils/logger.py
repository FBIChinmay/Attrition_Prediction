import logging
import os
from typing import Optional
import sys

def setup_logger(name: str, log_file: Optional[str] = "app.log", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and configures a logger.
    
    Args:
        name (str): Name of the logger.
        log_file (Optional[str]): File path to save logs. Defaults to "app.log".
        level (int): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured logger instance.
        
    Example:
        >>> logger = setup_logger("my_module")
        >>> logger.info("This is an info message")
    """
    try:
        logger = logging.getLogger(name)
        
        # If logger already has handlers, return it to avoid duplicate logs
        if logger.hasHandlers():
            return logger
            
        logger.setLevel(level)
        
        # Define formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        if log_file:
            # Ensure directory exists if path contains directories
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        return logger
    except Exception as e:
        print(f"Failed to setup logger: {str(e)}")
        # Fallback to a basic logger
        return logging.getLogger(name)
