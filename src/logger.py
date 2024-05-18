import logging
import os
import pathlib


class Logger(object):
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }

    def __init__(self, name=__name__, log_file="main.log", level="debug"):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level_relations.get(level))
        log_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.resolve(), "logs"
        )
        file_handler = logging.FileHandler(os.path.join(log_folder, log_file))
        formatter = logging.Formatter(
            "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()  # on-screen output
        console_handler.setFormatter(formatter)  # Setting the format

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_debug(self, message):
        self.logger.debug(message)


# Example usage:
if __name__ == "__main__":

    from logger import Logger

    logger = Logger(log_file="example.log", level="debug")
    logger.log_debug("This is a debug message")
    logger.log_info("This is an info message")
    logger.log_error("This is an error message")
    logger.log_warning("This is a warning message")
