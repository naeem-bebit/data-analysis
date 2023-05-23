import logging
import mylib
import pandas as pd
df = pd.DataFrame({'col1': [2, 1], 'col2': [4, 1]})

logging.basicConfig(
    filename='myapp.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('my_logger')


def main():

    logger.info('Started')
    mylib.do_something()
    logger.info('Finished')
    logger.info(df)


if __name__ == '__main__':
    main()
