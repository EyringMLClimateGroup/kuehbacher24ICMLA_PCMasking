"""
Learning rate schedule

Created on 2019-01-28-13-25
Author: Stephan Rasp, raspstephan@gmail.com
"""

import numpy as np


class LRUpdate(object):
    def __init__(self, init_lr, step, divide):
        # From goo.gl/GXQaK6
        self.init_lr = init_lr
        self.step = step
        self.drop = 1.0 / divide

    def __call__(self, epoch):
        lr = self.init_lr * np.power(self.drop, np.floor((epoch) / self.step))
        print(f"\nLearning rate = {lr}\n")
        return lr
