import logging


RESET = "\033[0m"

COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",   # Green
    "WARNING": "\033[93m",# Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m" # Magenta
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_msg = super().format(record)
        return f"{COLORS.get(record.levelname, RESET)}{log_msg}{RESET}"

class LogginModule():

    def __init__(self, app_name:str=None, output_logging_file_name:str=None, level:str=logging.INFO):
        self.logger = logging.getLogger(app_name if app_name else __name__)
        self.logger.setLevel(level)
        self.console_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(f"{output_logging_file_name}/{app_name}_{__name__}.log" if output_logging_file_name else f"{app_name}_{__name__}.log" )
        self.console_handler.setLevel(level)
        self.file_handler.setLevel(level)
        self.logging_format()
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
    
    def logging_format(self):
        console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d')
        self.console_handler.setFormatter(console_formatter)
        self.file_handler.setFormatter(file_formatter)
    
    def get_logger(self):
        return self.logger



