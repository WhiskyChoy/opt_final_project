from MyProblem import MinimalSurfaceProblem
from typing import Callable, Union
import numpy as np
from Problem import Problem
from Solver import Solver
from overrides import overrides
from autoclass import autoargs
import matplotlib.pyplot as plt
from utils import draw_custom, draw_origin
from functools import reduce
import time


class MinimalSurfaceSolver(Solver):
    final_solution: list = None
    surface_points: list = None
    support_points: list = None
    
    f_value_arr: list = []
    g_norm_arr: list = []
    cpu_time_arr: list = []

    def draw_solved_gap(self):
        iter_numbers = len(self.f_value_arr)
        f_opt = self.f_value_arr[-1]
        gap_arr = np.abs(np.array(self.f_value_arr) -
                         np.tile(f_opt, iter_numbers)) / max(1, np.abs(f_opt))
        iter_arr = list(range(iter_numbers))

        fig1 = plt.figure('iter_vs_gap').add_subplot(111)
        fig1.plot(iter_arr, gap_arr)

        fig2 = plt.figure('time_vs_gap').add_subplot(111)
        fig2.plot(self.cpu_time_arr, gap_arr)

        fig1.set_title(
            'Number of Iteration vs Relative Objective Function Gap')
        fig2.set_title('Elapsed Cpu-Time vs Relative Objective Function Gap')

        fig1.set_xlabel('The Number of Iterations')
        fig1.set_ylabel('The Relative Objective Function Gap')

        fig2.set_xlabel('Elapsed Cpu-Time')
        fig2.set_ylabel('The Relative Objective Function Gap')

        plt.show()

    def draw_g_norm(self):
        iter_numbers = len(self.f_value_arr)
        iter_arr = list(range(iter_numbers))

        fig1 = plt.figure('iter_vs_gradient').add_subplot(111)
        fig1.plot(iter_arr, self.g_norm_arr)

        fig2 = plt.figure('time_vs_gradient').add_subplot(111)
        fig2.plot(self.cpu_time_arr, self.g_norm_arr)

        fig1.set_title('Number of Iteration vs Norm of Gradient')
        fig2.set_title('Elapsed Cpu-Time vs Norm of Gradient')

        fig1.set_xlabel('The Number of Iterations')
        fig1.set_ylabel('The Norm of Gradient')

        fig2.set_xlabel('Elapsed Cpu-Time')
        fig2.set_ylabel('The Norm of Gradient')

    def show_summary(self, log: bool = True):
        exp = \
            f'The iteration number is: {len(self.f_value_arr)-1}\n' + \
            f'The final objective function value is: {round(self.f_value_arr[-1],5)}\n' + \
            f'The final norm of the gradient is: {round(self.g_norm_arr[-1],5)}\n' + \
            f'Cost time: {round(self.cpu_time_arr[-1] - self.cpu_time_arr[0], 3)} seconds\n'
        if log:
            print(exp)
        return exp

    def draw_3d(self, use_custom: bool=False, colorful: bool=False):
        if use_custom:
            draw_custom(self.surface_points, self.support_points)
        else:
            draw_origin(self.surface_points, self.support_points, colorful)

    def reset(self):
        self.final_solution = None
        self.surface_points = None
        self.support_points = None

        self.f_value_arr = []
        self.g_norm_arr = []
        self.cpu_time_arr = []


class GradientArmijoSolver(MinimalSurfaceSolver):
    @autoargs
    def __init__(self, tol=1e-3, max_iter=10000, s=1, sigma=0.5, gamma=0.1):
        pass

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        start = time.time()
        # Step1: calculate the gradient and its norm named 'ng'
        vars_current = init_vars
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        objv_current = problem.aim_func(vars_current)
        # Record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        # Step2: Iteration
        iter_count = 0
        while ng > self.tol and iter_count < self.max_iter:
            current = time.time()
            # INNER STEP1: get d_k
            d_k = -grad_value
            # INNER STEP2: get a_k
            alpha = self.s  # initial alpha as self.s
            objv_next = problem.aim_func(vars_current + alpha * d_k)
            while np.isnan(objv_next) or ((objv_next - objv_current) > (self.gamma * alpha * grad_value @ d_k)):
                alpha *= self.sigma
                objv_next = problem.aim_func(vars_current + alpha * d_k)
            vars_current = vars_current + alpha * d_k  # Update vars
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            # Record
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)

        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)
        return vars_current, grad_value, objv_current, ng, iter_count


class GlobalizedNewtonSolver(MinimalSurfaceSolver):
    @autoargs
    def __init__(self, tol=1e-3, max_iter=10000, s=1, sigma=0.1, gamma=0.1,
                 beta1=1e-6, beta2=1e-6, p=0.1):
        pass

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)        
        self.reset()
        start = time.time()
        vars_current = init_vars
        objv_current = problem.aim_func(vars_current)
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        # record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        iter_count = 0
        while ng > self.tol and iter_count < self.max_iter:
            current = time.time()
            # INNER STEP1: get d_k, use s_k or -grad_value
            hessian = problem.cal_hessian(vars_current)
            s_k = None
            s_k_available = False
            if(np.linalg.det(hessian) >= 1e-6):  # invertible
                s_k = -np.linalg.inv(hessian) @ grad_value
                if -grad_value @ s_k >= min(self.beta1, self.beta2 * np.linalg.norm(s_k, self.p)) * np.linalg.norm(s_k) ** 2:
                    s_k_available = True
            if s_k_available:
                d_k = s_k
            else:
                d_k = -grad_value
            # INNER STEP2: get a_k
            alpha = self.s  # initial alpha as self.s
            objv_next = problem.aim_func(vars_current + alpha * d_k)
            while np.isnan(objv_next) or ((objv_next - objv_current) > (self.gamma * alpha * grad_value @ d_k)):
                alpha *= self.sigma
                objv_next = problem.aim_func(vars_current + alpha * d_k)
            vars_current = vars_current + alpha * d_k
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            # record
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)

        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)
        return vars_current, grad_value, objv_current, ng, iter_count

class LBFGSSolver(MinimalSurfaceSolver):
    @autoargs
    def __init__(self, tol=1e-3, max_iter=10000, s=1, sigma=0.5, gamma=0.1, m=10):
        pass

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        # print('Backtracking')
        start = time.time()
        # Step1: calculate the gradient and its norm named 'ng'
        vars_current = init_vars
        paths = [init_vars]
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        ng_list = [np]
        # Step2: Compute the first iteration
        s_list = []
        y_list = []
        rho_list = []
        hessian = np.identity(len(grad_value))
        d_k = -hessian @ grad_value
        alpha = self.s  # initial alpha as self.s
        objv_current = problem.aim_func(vars_current)
        ObjV_temp = problem.aim_func(vars_current + alpha * d_k)
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        while np.isnan(ObjV_temp) or ((ObjV_temp - objv_current) > (self.gamma * alpha * grad_value @ d_k)):
            alpha *= self.sigma
            ObjV_temp = problem.aim_func(vars_current + alpha * d_k)
        s_list.append(alpha * d_k)
        y_list.append(problem.cal_gradient(vars_current + alpha * d_k) - problem.cal_gradient(vars_current))
        rho_list.append(1 / (s_list[-1] @ y_list[-1]))
        vars_current = vars_current + alpha * d_k
        objv_current = problem.aim_func(vars_current)
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        ng_list.append(np.log(ng))
        paths.append(vars_current)
        # Step3: Iteration
        iter_count = 1
        while ng > self.tol and iter_count < self.max_iter:
            current = time.time()
            # Compute d_k
            q = problem.cal_gradient(vars_current)
            objv_current = problem.aim_func(vars_current)
            a_list = []
            for i in range(min(self.m, len(s_list))):
                a = rho_list[-(i+1)] * (s_list[-(i+1)] @ q)
                a_list.append(a)
                q = q - a * y_list[-(i+1)]
            gamma = (s_list[-1] @ y_list[-1])/((y_list[-1] @ y_list[-1]) + 1e-8)
            Hessian = gamma * np.identity(len(q))
            r = Hessian @ q
            for i in range(min(self.m, len(s_list))):
                beta = rho_list[i] * (y_list[i] @ r)
                r = r + (a_list[-(i+1)] - beta) * s_list[i]
            d_k = -r
            # Compute alpha
            alpha = self.s  # initial alpha as self.s
            ObjV_temp = problem.aim_func(vars_current + alpha * d_k)
            while np.isnan(ObjV_temp) or (ObjV_temp - objv_current > self.gamma * alpha * (- ng**2)):
                alpha *= self.sigma
                ObjV_temp = problem.aim_func(vars_current + alpha * d_k)
            s_list.append(alpha * d_k)
            y_list.append(problem.cal_gradient(vars_current + alpha * d_k) - problem.cal_gradient(vars_current))
            rho_list.append(1 / (s_list[-1] @ y_list[-1]))
            if len(s_list) > 10:
                del s_list[0]
                del y_list[0]
                del rho_list[0]
            vars_current = vars_current + alpha * d_k  # Update init_vars
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            ng_list.append(ng)
            paths.append(vars_current)
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
        # output
        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)

        return vars_current, grad_value, objv_current, ng, iter_count

class InertialTechniqueSolver(MinimalSurfaceSolver):
    @autoargs
    def __init__(self, tol=1e-3, max_iter=10000, s=1, sigma=0.5, gamma=0.1):
        pass

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        start = time.time()
        # Step1: calculate the gradient and its norm named 'ng'
        vars_current = init_vars
        vars_before = vars_current
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        # ng_seq = []
        # Step2: Iteration
        objv_current = problem.aim_func(vars_current)
        # Record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        iter_count = 0
        while ng > self.tol and iter_count < self.max_iter:
            current = time.time()
            #choose thetak = 2/(k+2)
            if iter_count >= 2:
                beta_k = (iter_count -1)/(iter_count +2)
            else: 
                beta_k = 0

            y_k = vars_current + beta_k*(vars_current-vars_before) 
            grad_value = problem.cal_gradient(y_k)

            print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
            # INNER STEP1: get d_k
            d_k = -grad_value
            # INNER STEP2: get a_k
            alpha = self.s  # initial alpha as s
            objv_current = problem.aim_func(y_k)
            objv_next = problem.aim_func(y_k + alpha * d_k)
            while np.isnan(objv_next) or ((objv_next - objv_current) > (self.gamma * alpha * grad_value @ d_k)):
                alpha *= self.sigma
                objv_next = problem.aim_func(y_k + alpha * d_k)
 
            vars_next  = y_k-  alpha*problem.cal_gradient(y_k)
            vars_before = vars_current
            vars_current = vars_next
            # vars_current = vars_current + alpha * d_k  # Update vars
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            # ng_seq.append(ng)
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)
        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)

        return vars_current, grad_value, objv_current, ng, iter_count

class BarzilaiBorweinSolver(MinimalSurfaceSolver):
    @autoargs
    def __init__(self, tol=1e-3, max_iter=10000, s=1, sigma=0.5, gamma=0.1):
        pass

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        start = time.time()
        # Step1: calculate the gradient and its norm named 'ng'
        vars_current = init_vars
        vars_before = vars_current
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        grad_before = grad_value
        # ng_seq = []
        # Step2: Iteration
        objv_current = problem.aim_func(vars_current)
        iter_count = 0
        # Record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        while ng > self.tol and iter_count < self.max_iter:
            current = time.time()
            # INNER STEP1: get d_k
            d_k = -grad_value
            # INNER STEP2: get a_k
            alpha = self.s  # initial alpha as s
            objv_next = problem.aim_func(vars_current + alpha * d_k)
            if iter_count == 0:
                while np.isnan(objv_next) or ((objv_next - objv_current) > (self.gamma * alpha * grad_value @ d_k)):
                    alpha *= self.sigma
                    objv_next = problem.aim_func(vars_current + alpha * d_k)
            else:
                sk = vars_current-vars_before                
                yk = grad_value - grad_before
                alpha = sk.T@yk/(yk.T@yk)
                print(alpha)
            grad_before = grad_value
            vars_before = vars_current
            vars_current = vars_current + alpha * d_k  # Update vars
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            # print(grad_value,grad_before)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            # ng_seq.append(ng)
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)
        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)

        return vars_current, grad_value, objv_current, ng, iter_count

class ExactLineSearchSolver(MinimalSurfaceSolver):
    @autoargs
    def __init__(self, tol=1e-3, max_iter=100000, s=1, sigma=0.5, gamma=0.1):
        pass

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        start = time.time()
        # Step1: calculate the gradient and its norm named 'ng'
        vars_current = init_vars
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        hessian = problem.cal_hessian(vars_current)
        # ng_seq = []
        # Step2: Iteration
        objv_current = problem.aim_func(vars_current)
        iter_count = 0
        # Record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        while ng > self.tol and iter_count < self.max_iter:
            current = time.time()
            # INNER STEP1: get d_k
            d_k = -grad_value
            # INNER STEP2: get a_k
            #alpha = self.s  # initial alpha as s
            alpha = -grad_value.T*d_k/(d_k.T@hessian@d_k)
            print(alpha)
            vars_current = vars_current + alpha * d_k  # Update vars
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            # ng_seq.append(ng)
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)
        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)
        return vars_current, grad_value, objv_current, ng, iter_count

def project(vars: np.ndarray, b_s: Callable=None):
    if b_s is None:
        return vars
    else:
        return np.maximum(vars, b_s(vars))

class ProjectedGradientArmijoSolver(GradientArmijoSolver):
    def __init__(self, tol=1e-3, max_iter=10000, s=1, sigma=0.1, gamma=0.1, lambda_val: Union[int, Callable] = 1):
        super().__init__(tol, max_iter, s, sigma, gamma)
        if isinstance(lambda_val, int):
            def lambda_at(_): return lambda_val
        else:
            lambda_at = lambda_val

        self.lambda_at = lambda_at

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        start = time.time()
        # Step1: calculate the gradient and its norm named 'ng'
        vars_current = init_vars
        grad_value = problem.cal_gradient(vars_current)
        ng = np.linalg.norm(grad_value)
        objv_current = problem.aim_func(vars_current)
        # Record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        # Step2: Iteration
        iter_count = 0
        while iter_count < self.max_iter:
            current = time.time()
            # INNER STEP1: get d_k
            lambda_val = self.lambda_at(iter_count)
            d_k = project(vars_current-lambda_val*grad_value, problem.b_s)-vars_current
            d_k_norm = np.linalg.norm(d_k)
            if d_k_norm <= lambda_val * self.tol:
                break
            # INNER STEP2: get a_k
            alpha = self.s  # initial alpha as self.s
            objv_next = problem.aim_func(vars_current + alpha * d_k)
            while np.isnan(objv_next) or ((objv_next - objv_current) > (self.gamma * alpha * grad_value @ d_k)):
                alpha *= self.sigma
                objv_next = problem.aim_func(vars_current + alpha * d_k)
            vars_current = vars_current + alpha * d_k  # Update vars
            objv_current = problem.aim_func(vars_current)
            grad_value = problem.cal_gradient(vars_current)
            ng = np.linalg.norm(grad_value)
            iter_count += 1
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            # Record
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)

        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)
        self.support_points = problem.get_all_support_points()
        return vars_current, grad_value, objv_current, ng, iter_count

# TODO The penalty method does not work
class PenaltyMethodSolver(MinimalSurfaceSolver):

    def __init__(self, tol=1e-3, max_iter=100, inner_tol=1e-2, inner_iter_max=50, s=1, sigma=0.1, gamma=0.1, penalty_s=0.0001, penalty_gamma=10):
        self.penalty_s = penalty_s
        self.penalty_gamma = penalty_gamma
        self.tol = tol
        self.max_iter = max_iter
        self.solver = GradientArmijoSolver(inner_tol, inner_iter_max, s, sigma, gamma)

    @overrides
    def solve(self, problem: Problem, init_vars: np.ndarray=None, show_process:bool = False):
        if init_vars is None:
            init_vars = np.random.rand(problem.dim)
        self.reset()
        start = time.time()

        def aim_func(penalty_alpha: float, vars: np.ndarray) -> float:
             g_vals = problem.b_s(vars) - vars
             return problem.aim_func(vars) + 1/2 * penalty_alpha * np.sum(np.maximum(0, g_vals)**2)

        def cal_gradient(penalty_alpha: float, vars: np.ndarray) -> float:
            g_vals = problem.b_s(vars) - vars
            return problem.cal_gradient(vars) - penalty_alpha * np.sum(np.maximum(0, g_vals))

        class PenaltyProblem(MinimalSurfaceProblem):
            def __init__(self, penalty_alpha) -> None:
                self.penalty_alpha = penalty_alpha

            @overrides
            def aim_func(self, vars: np.ndarray):
                return aim_func(self.penalty_alpha, vars)

            @overrides
            def cal_gradient(self, vars: np.ndarray):
                return cal_gradient(self.penalty_alpha, vars)

            @overrides
            def cal_hessian(*_):
                pass

            @overrides
            def get_all_surface_points(*_):
                pass

        penalty_alpha = self.penalty_s
        current_problem = PenaltyProblem(penalty_alpha)
        vars_current, grad_value, objv_current, ng, _ = self.solver.solve(current_problem, init_vars)
        ng = np.linalg.norm(grad_value)
        # Record
        self.cpu_time_arr.append(0)
        self.f_value_arr.append(objv_current)
        self.g_norm_arr.append(ng)
        iter_count = 0
        while iter_count < self.max_iter:
            current = time.time()
            current_problem = PenaltyProblem(penalty_alpha)
            vars_current, grad_value, objv_current, ng, _ = self.solver.solve(current_problem, vars_current)
            penalty_alpha *= self.penalty_gamma
            iter_count += 1
            if reduce(np.logical_and, vars_current < problem.b_s(vars_current)):
                break
            if show_process:
                print(f'iteration {iter_count} : {round(ng, 8)}', end=' ')
                print(f'time: {round(time.time() - current, 3)} second(self.s)')
            # Record
            self.cpu_time_arr.append(time.time()-start)
            self.f_value_arr.append(objv_current)
            self.g_norm_arr.append(ng)

        self.final_solution = vars_current
        self.surface_points = problem.get_all_surface_points(vars_current)
        self.support_points = problem.get_all_support_points()
        return vars_current, grad_value, objv_current, ng, iter_count