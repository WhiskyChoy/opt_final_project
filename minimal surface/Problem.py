from abc import ABCMeta, abstractmethod
import numpy as np

class Problem(metaclass=ABCMeta):
    
    @abstractmethod
    def aim_func(self, vars: np.ndarray):
        pass

    @abstractmethod
    def cal_gradient(self, vars: np.ndarray):
        pass

    @abstractmethod
    def cal_hessian(self, vars: np.ndarray):
        pass