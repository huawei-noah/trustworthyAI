import os
import pathlib
import logging

_logger = logging.getLogger(__name__)


def create_dir(dir_path):
    """
    Create directory if it does not exist
    """
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        _logger.critical('Creating directories error: {0}'.format(err))
        exit(-1)