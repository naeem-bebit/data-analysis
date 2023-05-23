import logging

# Create a root logger
logging.basicConfig(level=logging.DEBUG)

# Create a logger for Module A
logger_a = logging.getLogger('module_a')
logger_a.setLevel(logging.DEBUG)

# Create a logger for Module B
logger_b = logging.getLogger('module_b')
logger_b.setLevel(logging.WARNING)

# Create a file handler for Module A
handler_a = logging.FileHandler('module_a.log')
handler_a.setLevel(logging.DEBUG)

# Create a console handler for Module B
handler_b = logging.StreamHandler()
handler_b.setLevel(logging.WARNING)

# Create formatters
formatter_a = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter_b = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Assign handlers to loggers
logger_a.addHandler(handler_a)
logger_b.addHandler(handler_b)

# Set formatters for handlers
handler_a.setFormatter(formatter_a)
handler_b.setFormatter(formatter_b)

# Log messages using the loggers
logger_a.debug('This is a debug message from Module A')
logger_a.info('This is an informational message from Module A')

logger_b.warning('This is a warning message from Module B')
logger_b.error('This is an error message from Module B')
