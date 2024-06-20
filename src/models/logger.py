import logging
import os
import pathlib


class Logger:
    # Mapping string level names to logging module constants
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }

    def __init__(self, name=__name__, log_file="main.log", level="debug"):

        # Initialize the logger with the given name and level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            self.level_relations.get(level, logging.DEBUG)
        )  # Default to DEBUG if level is invalid

        # Set up log folder path
        log_folder = pathlib.Path(__file__).parent.parent.resolve() / "logs"
        log_folder.mkdir(exist_ok=True)  # Create the log folder if it doesn't exist

        # Check if the logger already has handlers
        if not self.logger.handlers:
            # Create file handler for logging to a file
            file_handler = logging.FileHandler(log_folder / log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Create console handler for logging to the console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log_info(self, message: str):
        """Logs an info message."""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Logs a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Logs an error message."""
        self.logger.error(message)

    def log_debug(self, message: str):
        """Logs a debug message."""
        self.logger.debug(message)


# Example usage:
if __name__ == "__main__":

    from logger import Logger

    logger = Logger(log_file="example.log", level="debug")
    logger.log_debug("This is a debug message")
    logger.log_info("This is an info message")
    logger.log_error("This is an error message")
    logger.log_warning("This is a warning message")
