import os
from typing import Dict
import json
import inspect
import logging


class Logger:
    def __init__(self, log_files: str=None, log_level=None) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Handler for saving logs
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s => %(message)s')
        file_handler = logging.FileHandler(log_files, mode='a')

        # Handler for showing logs in terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def __log(self, level, message):

        frame = inspect.currentframe().f_back.f_back
        class_name = frame.f_locals['self'].__class__.__name__ if 'self' in frame.f_locals else ""
        function_name = frame.f_code.co_name

        full_message = f"[{class_name} : {function_name}] = {level}->{message}"

        if level == "debug":
            self.logger.debug(full_message)
        elif level == "info":
            self.logger.info(full_message)
        elif level == "error":
            self.logger.error(full_message)
        elif level == "critical":
            self.logger.critical(full_message)
        else:
            self.logger.warning(full_message)

    def debug(self, message: str):
        self.__log('debug', message)


    def info(self, message: str):
        self.__log('info', message)


    def error(self, message: str):
        self.__log('error', message)


    def critical(self, message: str):
        self.__log('critical', message)

    def warning(self, message: str):
        self.__log('warning', message)


LOG = Logger(log_files="logs.log", log_level=logging.DEBUG)

def load_json_file(json_file_src: str) -> Dict[str, any]:
    """
    Load json config file
    :param json_file_src: relative address of json config file
    :return: Dictionary (key - value) key: string(parameter name) and value: anything
    """
    try:
        with open(os.path.join(os.getcwd(), json_file_src), mode="r") as jsfile:
            js_data = json.load(jsfile)
        return js_data
    except Exception as error:
        LOG.error(f"{error}")