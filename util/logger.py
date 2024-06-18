import logging
import os

def get_logger(log_file, log_level=logging.INFO):
    """
    Set up a logger that writes to a file with a specific format.
    
    Parameters:
    log_file (str): The path to the log file.
    log_level (logging.LEVEL): The logging level.

    Returns:
    logger: Configured logger instance.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('MLProject')
    logger.setLevel(log_level)
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
