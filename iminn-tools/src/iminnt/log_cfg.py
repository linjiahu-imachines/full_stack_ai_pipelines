import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if len(logger.handlers) == 0:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    logger.propagate = False


def set_log_file(path):
    file_handler = logging.FileHandler(path, mode="a")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    file_handler.setLevel(logger.level)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    logger.addHandler(file_handler)