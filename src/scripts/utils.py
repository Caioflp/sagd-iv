"""General purpose utilities.

"""
import logging
import os
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Callable


def setup_logger() -> None:
    """ Performs basic logging configuration.
    """
    logging.getLogger("src").setLevel(logging.INFO)

    file = logging.FileHandler(filename="run.log", mode="w")
    file.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    file.setFormatter(file_formatter)
    logging.getLogger("src").addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(console_formatter)
    logging.getLogger("src").addHandler(console)


# Decorator with arguments syntax is weird
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
def experiment(path: str = None, benchmark=False) -> Callable:
    """Marks a function as the main function of an experiment.

    Creates a separate directory for it to be run.

    """
    output_dir = Path("./outputs").resolve()
    now = datetime.now()
    if path is None:
        inner_output_dir = output_dir / now.strftime("%Y-%m-%d")
    else:
        inner_output_dir = output_dir / path.lower().replace(" ", "_")
    inner_output_dir.mkdir(parents=True, exist_ok=True)
    if benchmark:
        run_dir = inner_output_dir
    else:
        run_dir = inner_output_dir / ("run_" + str(len(os.listdir(inner_output_dir))))
        run_dir.mkdir(parents=True, exist_ok=False)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.chdir(run_dir)
            setup_logger()
            result = func(*args, **kwargs)
            os.chdir(output_dir)
            return result
        return wrapper
    return decorator

