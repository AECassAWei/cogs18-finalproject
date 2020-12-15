"""Classes used throughout project"""

import sys, os # suppress 'print' in stat_test

### ===================== Class to Suppress Print =================== ###
# This part of the code is adapted from the following website on Stack Overflow:
# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print/8391735#8391735
# This class is used to suppress print statement in stat_test function so that
# the format in the jupyter notebook won't be really messy

class SuppressPrintStatement:
    """
    A class used to suppress print messages.
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

### ===================== Class to Suppress Print =================== ###
