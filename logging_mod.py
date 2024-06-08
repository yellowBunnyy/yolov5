import logging

# Create a custom logger
logger = logging.getLogger('yolov5_logger')

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('my_logger.log')

# Set level for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example usage
# logger.warning('This will appear in the console')
# logger.error('This will appear in both the console and the log file')