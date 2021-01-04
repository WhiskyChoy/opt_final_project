from typing import Callable
import numpy as np
import MyProblem
import MySolver
from math import pi, sin, exp, cos, sqrt, asin
from random import random
from utils import ObstacleGenerator, draw_multiple_g_norm_iter, draw_multiple_solved_gap_iter, draw_multiple_solved_gap_cpu, draw_multiple_g_norm_cpu, draw_boundary_only, get_cie_arr
import cv2

# set the scale of our problem
n_arr = [5, 7, 9,13,15,18]

# define all r functions


def r_0(x, _): return 1+sin(2*pi*x)
def r_1(x, _): return 1+cos(1/(x+0.001))
def r_2(_, y): return 1/2 - np.abs(y - 1/2)
def r_3(x, y): return 1/(1+exp(x*y))
def r_4(x, y): return 1 + asin(-1+2*sqrt(x*y))

def r_np_0(x, _): return 1+np.sin(2*pi*x)
def r_np_1(x, _): return 1+np.cos(1/(x+0.001))
def r_np_2(_, y): return 1/2 - np.abs(y - 1/2)
def r_np_3(x, y): return 1/(1+np.exp(x*y))
def r_np_4(x, y): return 1 + np.arcsin(-1+2*np.sqrt(x*y))

# put the r fuctions inside an arr
r_arr = [r_0, r_1, r_2, r_3, r_4]


def get_constant_r(c: int = 0) -> Callable:
    def r(*_): return c
    return r


# init obstacle generator arr
obstacle_gen_arr = [ObstacleGenerator(n-2) for n in n_arr]

# init all solvers
solver_0 = MySolver.GradientArmijoSolver()
solver_1 = MySolver.GlobalizedNewtonSolver()
solver_2 = MySolver.LBFGSSolver()
solver_3 = MySolver.ExactLineSearchSolver()
solver_4 = MySolver.BarzilaiBorweinSolver()
solver_5 = MySolver.InertialTechniqueSolver()


solver_2_1 = MySolver.PenaltyMethodSolver()
solver_2_2 = MySolver.ProjectedGradientArmijoSolver()

# put all sovlers into one arr
unconstraint_solver_arr = [solver_0, solver_1, solver_2, solver_3, solver_4]

constraint_solver_arr = [solver_2_1 ,solver_2_2]

# init the problems using r_0 with different n
# problem_r_0_arr = [MyProblem.MSSymbolicEvalProblem(n, r_0) for n in n_arr]

# init the problems using different r with n=7
# problem_n_7 = [MyProblem.MSSymbolicEvalProblem(7, r) for r in r_arr]

# Experiment: compare autograd and symbolic


def exp_1():
    problem_auto_grad = MyProblem.MSAutoGradProblem(9, r_0)
    problem_symbolic = MyProblem.MSSymbolicEvalProblem(9, r_0)
    init_vars = np.random.rand(problem_auto_grad.dim)
    solver_1.solve(problem_auto_grad, init_vars, show_process=False)
    solver_1.show_summary()
    solver_2.solve(problem_symbolic, init_vars, False)
    solver_2.show_summary()

# draw the solution of problems with all r and n=7


# def exp_2():
#     for problem in problem_n_7:
#         solver_0.solve(problem)
#         solver_0.draw_3d()
def exp_3():
    n = 7
    obstacle_generator = ObstacleGenerator(n-2)
    obstacle_generator.add_unit_rect()
    obstacle_generator.add_point()
    problem = MyProblem.MSSymbolicEvalProblem(n, get_constant_r(1), obstacle_generator.obstacle, False)
    solver_0.solve(problem, show_process=True)
    solver_0.draw_solved_gap()
    solver_0.draw_g_norm()
    solver_0.draw_3d()

def exp_4():
    n = 7
    problem = MyProblem.MSSymbolicEvalProblem(n, r_2)
    init_vars = np.ones(problem.dim) * 2
    for solver in unconstraint_solver_arr:
        solver.max_iter = 200
        solver.solve(problem, init_vars=init_vars, show_process=True)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr, log_y=True)
    for solver in unconstraint_solver_arr:
        solver.show_summary()
        solver.draw_3d(colorful=True)

def exp_5():
    n = 7
    problem = MyProblem.MSSymbolicEvalProblem(n, r_2)
    init_vars = np.ones(problem.dim)
    for solver in unconstraint_solver_arr:
        solver.max_iter = 200
        solver.solve(problem, init_vars=init_vars, show_process=True)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr, log_x=True)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr, log_x=True)
    for solver in unconstraint_solver_arr:
        solver.show_summary()
        solver.draw_3d(colorful=True)

def exp_6():
    n = 7
    problem = MyProblem.MSSymbolicEvalProblem(n, r_2)
    init_vars = np.ones(problem.dim)
    for solver in unconstraint_solver_arr:
        solver.max_iter = 20
        solver.solve(problem, init_vars=init_vars, show_process=True)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr, log_x=True)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr, log_x=True)
    for solver in unconstraint_solver_arr:
        solver.show_summary()
        solver.draw_3d(colorful=True)

def exp_7():
    n = 17
    problem = MyProblem.MSSymbolicEvalProblem(n, r_2)
    init_vars = np.ones(problem.dim) * 2
    for solver in unconstraint_solver_arr:
        # solver.max_iter = 2000
        solver.solve(problem, init_vars=init_vars, show_process=True)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr)
    draw_multiple_solved_gap_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_solved_gap_cpu(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_g_norm_iter(solvers=unconstraint_solver_arr, log_y=True)
    draw_multiple_g_norm_cpu(solvers=unconstraint_solver_arr, log_y=True)
    for solver in unconstraint_solver_arr:
        solver.show_summary()
        solver.draw_3d(colorful=True)

def exp_8():
    n = 23
    obstacle = get_cie_arr(1)
    problem = MyProblem.MSSymbolicEvalProblem(n, r_2, obstacle=obstacle, need_hessian=False)
    init_vars = np.ones(problem.dim) * 2
    solver_2_2.solve(problem, init_vars, show_process=True)
    solver_2_2.draw_3d()
    solver_2_2.show_summary()

exp_arr = [exp_1]


def do_all_experiments():
    for do_experiment in exp_arr:
        do_experiment()


if __name__ == '__main__':
    # exp_7()
    # draw_boundary_only(r_np_0)
    # draw_boundary_only(r_np_1)
    # draw_boundary_only(r_np_2)
    # draw_boundary_only(r_np_3)
    # draw_boundary_only(r_np_4)
    exp_8()