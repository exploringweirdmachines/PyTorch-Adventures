USE_RECURSIVE = False

### Add as Module to Import ###
import sys
sys.modules[__name__].USE_RECURSIVE = USE_RECURSIVE

if USE_RECURSIVE:
    from .recursive_tensor import *
else:
    from .tensor import *

from .nn import *
from .optim import *
from .utils import * 
from .save_load import save, load