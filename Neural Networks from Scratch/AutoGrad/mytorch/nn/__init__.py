# Grab flag from mytorch/__init__.py
from .. import USE_RECURSIVE

if USE_RECURSIVE:
    print("Using Recursive Backwards")
    from .recursive_functional import *
    from .recursive_modules import *

else:
    print("Using Topological Sort Backwards")
    from .functional import *
    from .modules import *