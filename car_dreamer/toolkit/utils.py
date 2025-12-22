import os
import sys
from loguru import logger

_file_logging_config = None

def get_logger(log_dir, log_level="INFO", job_name="default"):
    """
    Create a logger.
    """
    global _file_logging_config
    
    if job_name in ["train", "eval"]:
        _file_logging_config = {"log_dir": log_dir, "job_name": job_name}

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.remove()
    logger.add(sys.stderr, level=log_level, format=log_format, colorize=True)

    if _file_logging_config is not None:
        config = _file_logging_config
        log_file_info = os.path.join(config["log_dir"], f"{config['job_name']}_info.log")
        log_file_debug = os.path.join(config["log_dir"], f"{config['job_name']}_debug.log")
        log_file_error = os.path.join(config["log_dir"], f"{config['job_name']}_error.log")

        logger.add(log_file_info, level="INFO", format=log_format, rotation="10 MB", compression="zip")
        logger.add(log_file_debug, level="DEBUG", format=log_format, rotation="10 MB", compression="zip")
        logger.add(log_file_error, level="ERROR", format=log_format, rotation="10 MB", compression="zip")
    
    return logger
