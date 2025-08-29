import logging

# Configure logging to show INFO level and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_logger")

if __name__ == "__main__":
    logger.info('test_info')