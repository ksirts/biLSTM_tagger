import logging
import sys
import os.path

def get_logger(fn, level='info'):
    '''
    create a logger and output to file and stdout
    '''
    assert level in ['info', 'debug']

    # Create a custom logger
    logger = logging.getLogger(__name__)
    level = {'info': logging.INFO, 'debug': logging.DEBUG}[level]
    logger.setLevel(level)

    fn = '.models/' + fn
    if os.path.exists(fn):
        os.remove(fn)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(fn, mode='a')

    # Create formatters and add them to handlers
    format = '%(asctime)s %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    c_format = logging.Formatter(fmt=format, datefmt=date_format)
    f_format = logging.Formatter(fmt=format, datefmt=date_format)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def safe_division(x, y):
    return 0.0 if y == 0 else x / y


def get_model_fn(model_name, index):
    model_fn = model_name
    if index is not None:
        model_fn += '-' + str(index)
    return model_fn
