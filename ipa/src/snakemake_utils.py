import logging
import os
from datetime import datetime




def _create_logger(path:str,name: str) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Args: 
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Logger
    """

    if not os.path.exists(path+'/logs/'):
        os.makedirs(path+'/logs/')

    logger = logging.Logger(name.capitalize())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{path}/logs/{name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger



def create_logger_format(p,wildcards):
    return _create_logger(p,f"{wildcards.filename}")
    
def create_logger_workflow(wildcards):
    return _create_logger(f"{wildcards.path}",f"{wildcards.filename}_method_{wildcards.method}_cxy_{wildcards.crop_sizexy}_cz_{wildcards.crop_size_z}")