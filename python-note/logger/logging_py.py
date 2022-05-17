import logging
import pandas as pd

# logging.basicConfig(level=logging.DEBUG, filename="test.log", filemode="w",
#                     format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('test.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("test custom logger")

a = 1
logger.info(f"This is {a}")

try:
    1/0
except ZeroDivisionError:
    logger.exception("Exception occured")

try:
    b
except NameError:
    logger.exception("THE ERROR")
# c
# logging.debug("debug")
# logging.info("info")
# logging.warning("warning")
# logging.error("error")
# logging.critical("critical")

df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, 2, 1, 8]},
                  index=['falcon', 'dog', 'spider', 'fish'])
# logger.info(df)
logger.info(f"\n {df}")#.to_string().replace('\n', '\n\t'))
logger.info(f"preprocess_blade \n {df}")