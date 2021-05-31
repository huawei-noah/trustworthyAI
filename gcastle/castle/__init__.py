__version__ = "1.0.1rc1"


import sys
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')
