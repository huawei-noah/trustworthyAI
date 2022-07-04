#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import glob
from os.path import dirname, basename, isfile
files = glob.glob(dirname(__file__) + '/*.py')
__all__ = [basename(f)[:-3] for f in files if isfile(f) and not f.endswith('__init__.py')]
from .mask import *
