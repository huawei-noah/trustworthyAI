import sys
import os
import pathlib
import logging

_logger = logging.getLogger(__name__)


def create_dir(dir_path):
    """
    dir_path - A path of directory to create if it is not found
    :param dir:
    :return exit_code: 0:success -1:failed
    """
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        _logger.critical('Creating directories error: {0}'.format(err))
        sys.exit(-1)
