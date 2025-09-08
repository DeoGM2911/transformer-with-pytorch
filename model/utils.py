# utils.py
#
# Utility functions for models.
#
# @author: Dung Tran
# @date: September 8, 2025

import inspect


def save_hyperparams(self: object):
    """
    Save the hyperparameters to the class instance.
    """
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    params = {k: v for k, v in local_vars.items() if k != 'self'}
    
    for k, v in params.items():
        if not self.__dict__.get(k, None):
            self.__setattr__(k, v)


class Test():
    def __init__(self, a, b, c=3):
        save_hyperparams(self)
    
    def show(self):
        print(self.a, self.b, self.c)
    
if __name__ == "__main__":
    t = Test(1, 2)
    t.show()  # Output: 1 2 3
    
    t = Test(4, 5, 2)
    t.show()  # Output: 4 5 2