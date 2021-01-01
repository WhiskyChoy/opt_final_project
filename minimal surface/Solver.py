from abc import ABCMeta, abstractmethod
from Problem import Problem
import numpy as np

class Solver(metaclass=ABCMeta):
    
    @abstractmethod
    def solve(self, problem: Problem, init_vars: np.ndarray):
        pass