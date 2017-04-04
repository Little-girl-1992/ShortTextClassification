from flask import Flask
import os
import configparser
import logging
import warnings

warnings.filterwarnings("ignore")


def create_app():
    app = Flask(__name__)

    logger = logging.getLogger('__main__')
    logger.setLevel(logging.WARNING)

    logger_info = logging.getLogger('info')
    logger_error = logging.getLogger('error')
    logger_time = logging.getLogger('time')
    logger_info.setLevel(logging.INFO)
    logger_error.setLevel(logging.WARNING)
    logger_time.setLevel(logging.INFO)

    cf = configparser.ConfigParser()
    cf.read(os.path.dirname(__file__) + '/main/initialization.conf', encoding='UTF-8')

    log_root = ''
    if 'logging' in cf.sections() and 'log_root' in cf.options('logging'):
        log_root = cf.get('logging', 'log_root')

    handler_error = logging.FileHandler(log_root + '/error.log', encoding='UTF-8')
    handler_info = logging.FileHandler(log_root + '/info.log', encoding='UTF-8')
    handler_time = logging.FileHandler(log_root + '/time.log', encoding='UTF-8')

    logging_format = logging.Formatter(
		'%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler_error.setFormatter(logging_format)
    handler_info.setFormatter(logging_format)
    handler_time.setFormatter(logging_format)

    logger_error.addHandler(handler_error)
    logger_info.addHandler(handler_info)
    logger_time.addHandler(handler_time)

    from .main import main as main_blueprint

    app.register_blueprint(main_blueprint)

    return app
