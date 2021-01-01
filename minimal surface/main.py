import numpy as np
import MyProblem
import MySolver
from math import pi, sin, exp, cos, sqrt, asin
from random import random
from utils import ObstacleGenerator

# set the scale of our problem
n_arr = [5, 7, 9]

# define all r functions
def r_0(x, _): return 1+sin(2*pi*x)
def r_1(x, _): return 1+cos(1/(x+0.001))
def r_2(_, y): return 1/2 - np.abs(y - 1/2)
def r_3(x, y): return 1/(1+exp(x*y))
def r_4(x, y): return 1 + asin(-1+2*sqrt(x*y))
def r_5(*_): return 0

# put the r fuctions inside an arr
r_arr = [r_0, r_1, r_2, r_3, r_4, r_5]

# init obstacle generator arr
obstacle_gen_arr = [ObstacleGenerator(n-2) for n in n_arr]

# init all solvers
solver_0 = MySolver.GradientArmijoSolver()
solver_1 = MySolver.GlobalizedNewtonSolver()
solver_2 = MySolver.LBFGSSolver()
solver_3 = MySolver.InertialTechniqueSolver()
solver_4 = MySolver.ExactLineSearchSolver()
solver_5 = MySolver.BarzilaiBorweinSolver()
solver_6 = MySolver.LBFGSSolver()
solver_7 = MySolver.PenaltyMethodSolver()
solver_8 = MySolver.ProjectedGradientArmijoSolver()

# put all sovlers into one arr
solver_arr = [solver_0, solver_1, solver_2, solver_3, solver_4, solver_5, solver_6, solver_7, solver_8]

# init the problems using r_0 with different n
# problem_r_0_arr = [MyProblem.MSSymbolicEvalProblem(n, r_0) for n in n_arr]

# init the problems using different r with n=7
problem_n_7 = [MyProblem.MSSymbolicEvalProblem(7, r) for r in r_arr]

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
def exp_2():
    for problem in problem_n_7:
        solver_1.solve(problem)
        solver_1.draw_3d()



exp_arr = [exp_1]

def do_all_experiments():
    for do_experiment in exp_arr:
        do_experiment()

if __name__ == '__main__':
    exp_2()
