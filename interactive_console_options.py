"""
A handful of settings that are useful for interactive work. Import the module at the console to activate them.
"""

import scipy as sp
import logging
import seaborn as sns

logging.getLogger().setLevel(0)
sp.set_printoptions(precision=3)
sns.set_context(rc={'figure.figsize': (10, 10)})