import numpy as np
import algopy
import time
from typing import Callable, Union
from Problem import Problem
from overrides import overrides
from sympy import symbols, lambdify, sqrt as sympy_sqrt, Matrix, Array
from math import sqrt, ceil

def get_cal_area_func(sqrt_func: Callable) -> Callable:

    def cal_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
            a = p1 - p2
            b = p1 - p3
            x1, y1, z1 = a[0], a[1], a[2]
            x2, y2, z2 = b[0], b[1], b[2]
            return sqrt_func((y1*z2-y2*z1)**2 + (z1*x2-z2*x1)**2 + (x1*y2-x2*y1)**2)

    return cal_area


def cal_algopy_cgraph(eval_f, var_num):
    cg = algopy.CGraph()
    x = algopy.Function(range(var_num))
    y = eval_f(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]
    return cg


def get_symbolic_exp(f: Callable, dim: int, need_hessian: bool):
    # Define the variables' symbol
    print('Start constructing the symbolic variables...')
    start = time.time()
    x = [0] * dim
    hessian_strs = ""
    for i in range(dim):
        hessian_strs += "x["+str(i)+"], "
    x = symbols(hessian_strs, real=True)
    end = time.time()
    print(f'Finished constructing the symbolic variables: {round(end - start, 3)} second(s)')

    print('Start constructing the symbolic function...')
    start = time.time()
    g = f(x)
    end = time.time()
    print(f'Finished constructing the symbolic function: {round(end - start, 3)} second(s)')

    # cal gradient
    print('Start calculating the gradient expression...')
    start = time.time()
    grad_expression = Matrix([g]).jacobian(x)
    hessian_expression = None
    end = time.time()
    print(f'Time cost in calculating gradient: {round(end - start, 3)} second(s)')

    if need_hessian:
        # cal hessian
        print('Start calculating the hessian expression...')
        start = time.time()
        # hessian_expression = hessian(g,x)   # It's too slow. We can use jacobian twice since the gradient is sparse
        hessian_expression = grad_expression.jacobian(x)
        end = time.time()
        print(f'Time cost in calculating hessian: {round(end - start, 3)} second(s)')

    return x, grad_expression, hessian_expression


def get_aim_func(n: int, get_height: Callable, cal_area: Callable) -> Callable:

    def aim_func(vars: np.ndarray) -> float:
        total_area = 0.0
        for i in range(n-1):
            for j in range(n-1):
                p1 = np.array([i/(n-1), j/(n-1),
                                get_height(vars, i, j)])
                p2 = np.array([i/(n-1), (j+1)/(n-1),
                                get_height(vars, i, j+1)])
                p3 = np.array([(i+1)/(n-1), j/(n-1),
                                get_height(vars, i+1, j)])
                p4 = np.array(
                    [(i+1)/(n-1), (j+1)/(n-1), get_height(vars, i+1, j+1)])
                area_1 = cal_area(p1, p2, p4)
                area_2 = cal_area(p1, p2, p3)
                total_area += area_1 + area_2
        return total_area

    return aim_func

class MinimalSurfaceProblem(Problem):
    def __init__(self, n: int, r: Callable, obstacle:Union[np.ndarray, Callable]=None) -> None:
        """
        n: int - Number of partition in length.
        """

        self.n = n  # n denotes the number of points of the edge
        self.dim = (n-2)**2
        self.r = r

        if obstacle is None:
            self.b = None
        else:
            self.obstacle = obstacle


        self._aim_func = get_aim_func(n, self.get_height, get_cal_area_func(np.sqrt))

    @property
    def obstacle(self):
        return self._obstacle

    @obstacle.setter
    def obstacle(self, input_obstacle:Union[np.ndarray, Callable]):
        if self.obstacle_is_valid(input_obstacle):
            self._obstacle = input_obstacle
            self.init_b()
        else:
            raise ValueError(f"obstacle.shape[0] should be {self.n-2}, and obstacle.shape[1] must be {self.n-2} or a callable function")

    def init_b(self):
        if isinstance(self.obstacle, Callable):
            def b(index: int) -> float:
                inner_row_length = self.n - 2
                i_inner = index // inner_row_length
                j_inner = index % inner_row_length
                i = i_inner + 1
                j = j_inner + 1
                return self.obstacle(i/(self.n-1), j/(self.n-1))
        else:
            def b(index: int) -> float:
                row_length = self.n - 2
                i = index // row_length
                j = index % row_length
                return self.obstacle[i][j]
        
        def b_s(vars: np.ndarray) -> np.ndarray:
            length = vars.shape[0]
            return np.array([b(i) for i in range(length)])
        
        self.b = b
        self.b_s = b_s

    def obstacle_is_valid(self, obstacle:Union[np.ndarray, Callable]):
        return (isinstance(obstacle, Callable) and obstacle.func_code.co_argcount==2) or (isinstance(obstacle, np.ndarray) and obstacle.shape[0] == self.n-2 and obstacle.shape[1] == self.n-2)

    @overrides
    def aim_func(self, vars: np.ndarray) -> float:
        return self._aim_func(vars)

    def get_height(self, vars: np.ndarray, i: int, j: int) -> float:
        """
        This function is utilized to get the correspond height of the specific location.
        """

        if 1 <= i <= self.n - 2 and 1 <= j <= self.n - 2:
            return vars[(self.n - 2) * (i - 1) + (j - 1)]
        else:
            return self.r(i/(self.n-1), j/(self.n-1))

    def get_all_surface_points(self, vars: np.ndarray) -> list:
        """
        This function is utilized to get the 3d locations of all points.
        """
        points = []
        for i in range(self.n):
            for j in range(self.n):
                points.append([i/(self.n-1), j/(self.n-1), self.get_height(vars, i, j)])
        return points

    def get_all_support_points(self) -> list:
        points = []
        if isinstance(self.obstacle, Callable):
            for i in range(self.n):
                for j in range(self.n):
                    points.append([i/(self.n-1), j/(self.n-1), self.obstacle(i/(self.n-1),j/(self.n-1))])
        else:
            for i in range(self.n-2):
                for j in range(self.n-2):
                    points.append([(i+1)/(self.n-1), (j+1)/(self.n-1), self.obstacle[i][j]])
        return points

class MSAutoGradProblem(MinimalSurfaceProblem):
    def __init__(self, n: int, r: Callable, obstacle:Union[np.ndarray, Callable]=None) -> None:
        super().__init__(n, r, obstacle)
        # calculate algopy_CGraph
        self.cg = cal_algopy_cgraph(self.aim_func, self.dim)

    @overrides
    def cal_gradient(self, vars: np.ndarray):
        return self.cg.gradient(vars.astype(float).tolist())[0]

    @overrides
    def cal_hessian(self, vars: np.ndarray):
        return self.cg.hessian(vars.astype(float).tolist())

class MSSymbolicLambdifyProblem(MinimalSurfaceProblem):
    def __init__(self, n: int, r: Callable, obstacle:Union[np.ndarray, Callable]=None, need_hessian: bool = True) -> None:
        super().__init__(n, r, obstacle)

        args, grad_expression, hessian_expression = get_symbolic_exp(
            get_aim_func(self.n, self.get_height, get_cal_area_func(sympy_sqrt)), self.dim, need_hessian)

        print('Start lambdifing the gradient expression...')
        start = time.time()
        self._cal_gradient = lambdify(args, grad_expression)
        end = time.time()
        end_sep = '' if need_hessian else '\n'
        print(f'time cost in lambdify gradient: {round(end - start, 3)} second(s){end_sep}')

        if need_hessian:
            print('Start lambdifing hessian expression...')
            start = time.time()
            self._cal_hessian = lambdify(args, hessian_expression)
            end = time.time()
            print(f'time cost in lambdify hessian: {round(end - start, 3)} second(s)\n')

    @overrides
    def cal_gradient(self, vars: np.ndarray):
        return self._cal_gradient(*vars)[0]

    @overrides
    def cal_hessian(self, vars: np.ndarray):
        return self._cal_hessian(*vars)


class MSSymbolicEvalProblem(MinimalSurfaceProblem):
    def __init__(self, n: int, r: Callable, obstacle:Union[np.ndarray, Callable]=None, need_hessian: bool = True) -> None:
        super().__init__(n, r, obstacle)

        _, grad_expression, hessian_expression = get_symbolic_exp(
            get_aim_func(self.n, self.get_height, get_cal_area_func(sympy_sqrt)), self.dim, need_hessian)
    
        print('Start evaling the gradient expression...')
        start = time.time()
        grad_strs = str(Array(grad_expression)[0])
        self._cal_gradient = eval('lambda x:' + grad_strs)
        end = time.time()
        end_sep = '' if need_hessian else '\n'
        print(f'time cost in evaling gradient: {round(end - start, 3)} second(s){end_sep}')

        if need_hessian:
            print('Start evaling hessian expression...')
            start = time.time()
            hessian_strs = str(Array(hessian_expression))
            self._cal_hessian = eval('lambda x:' + hessian_strs)
            end = time.time()
            print(f'time cost in evaling hessian: {round(end - start, 3)} second(s)\n')

        
    @overrides
    def cal_gradient(self, vars: np.ndarray):
        return np.array(self._cal_gradient(vars))

    @overrides
    def cal_hessian(self, vars: np.ndarray):
        return np.array(self._cal_hessian(vars))