import os
import sys
sys.path.append("..")
from utils.addd import addd

# -b 8
if __name__ == "__main__":
    
    print(os.getcwd())
    print(addd(12, 8))