import random
from typing import Callable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt, cm
import numpy as np
from colorsys import hsv_to_rgb
from math import sqrt
from random import randint, uniform

def draw_multiple_solved_gap_iter(solvers:list, log_x=False, log_y=False)->None:
    for solver in solvers:
        iter_numbers = len(solver.f_value_arr)
        f_opt = solver.f_value_arr[-1]
        gap_arr = np.abs(np.array(solver.f_value_arr) -
                         np.tile(f_opt, iter_numbers)) / max(1, np.abs(f_opt))
        iter_arr = list(range(iter_numbers))
        label = solver.name
        plt.plot(iter_arr, gap_arr, label=label)
    add_at_end = ' (log x)' if log_x else '' + ' (log y)' if log_y else ''
    plt.title(f'Number of Iteration vs Relative Objective Function Gap{add_at_end}')
    plt.xlabel('The Number of Iterations')
    plt.ylabel('The Relative Objective Function Gap')
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    plt.legend()
    plt.show()

def draw_multiple_solved_gap_cpu(solvers:list, log_x=False, log_y=False)->None:
    for solver in solvers:
        iter_numbers = len(solver.f_value_arr)
        f_opt = solver.f_value_arr[-1]
        gap_arr = np.abs(np.array(solver.f_value_arr) -
                         np.tile(f_opt, iter_numbers)) / max(1, np.abs(f_opt))
        label = solver.name
        plt.plot(solver.cpu_time_arr, gap_arr, label=label)
    add_at_end = ' (log x)' if log_x else '' + ' (log y)' if log_y else ''
    plt.title(f'Elapsed Cpu-Time vs Relative Objective Function Gap{add_at_end}')
    plt.xlabel('Elapsed Cpu-Time (seconds)')
    plt.ylabel('The Relative Objective Function Gap')
    if log_x:
        plt.xscale('symlog')
    if log_y:
        plt.yscale('log')
    plt.legend()
    plt.show()

def draw_multiple_g_norm_iter(solvers:list, log_x=False, log_y=False)->None:
    for solver in solvers:
        iter_numbers = len(solver.f_value_arr)
        iter_arr = list(range(iter_numbers))
        label = solver.name
        plt.plot(iter_arr, solver.g_norm_arr, label=label)
    add_at_end = ' (log x)' if log_x else '' + ' (log y)' if log_y else ''
    plt.title(f'Number of Iteration vs Norm of Gradient{add_at_end}')
    plt.xlabel('The Number of Iterations')
    plt.ylabel('The Norm of Gradient')
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    plt.legend()
    plt.show()

def draw_multiple_g_norm_cpu(solvers:list, log_x=False, log_y=False)->None:
    for solver in solvers:
        label = solver.name
        plt.plot(solver.cpu_time_arr, solver.g_norm_arr, label=label)
    add_at_end = ' (log x)' if log_x else '' + ' (log y)' if log_y else ''
    plt.title(f'Elapsed Cpu-Time vs Norm of Gradient{add_at_end}')
    plt.xlabel('Elapsed Cpu-Time (seconds)')
    plt.ylabel('The Norm of Gradient')
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    plt.legend()
    plt.show()

def get_color(val: float, max_val: float, base: float, change_range: float, inverse: bool)->tuple:
    hsv = None
    if not inverse:
        change_range = min(1-base, change_range)
        hsv = [base + change_range * val/max_val, 1, 1]
    else:
        change_range = min(base, change_range)
        hsv = [base - change_range * val/max_val, 1, 1]
    return hsv_to_rgb(*hsv)

def get_triangles(points: list) -> list:
    n = int(sqrt(len(points)))
    result = []
    for i in range(n-1):
        for j in range(n-1):
            p1 = points[i * n + j]
            p2 = points[i * n + (j+1)]
            p3 = points[(i+1) * n + j]
            p4 = points[(i+1) * n + (j+1)]
            result = [*result, [p1, p2, p4], [p1, p4, p3]]
    return result

def draw_origin(surface_points: list, inner_sup_points: list = None, colorful: bool = False, title:str=None):
    evaluate_points = [p[2] for p in surface_points]
    max_height = max(*evaluate_points)
    min_height = min(*evaluate_points)
    fig = plt.figure(title)
    ax = Axes3D(fig)
    surface_x, surface_y, surface_z = zip(*surface_points)
    surf = ax.plot_trisurf(surface_x, surface_y, surface_z, linewidth=0.2, antialiased=True, alpha=0.9, cmap=(cm.rainbow if colorful else None))
    # Add a color bar which maps values to colors.
    if colorful:
        fig.colorbar(surf, shrink=0.5, aspect=5)
    if inner_sup_points is not None and len(inner_sup_points)>0:
        # minius a little bit in height
        inner_sup_points = [[p[0], p[1], p[2]-1e-2] for p in inner_sup_points]
        support_x, support_y, support_z = zip(*inner_sup_points)
        ax.plot_trisurf(support_x, support_y, support_z, linewidth=0.2, antialiased=True, color='red')
    ax.set_zlim3d(min_height, 1.25 * max_height)
    ax.set_title(title,  loc='center')
    # plt.legend()
    plt.show()

def draw_custom(surface_points: list, inner_sup_points: list = None, surface_base: float=0.8, change_range: float=0.1, inverse: bool=True, support_color: float = 0.0, title:str=None):
    evaluate_points = [p[2] for p in surface_points]
    max_height = max(*evaluate_points)
    min_height = min(*evaluate_points)
    fig = plt.figure(title)
    ax = Axes3D(fig)

    surface_triangles = get_triangles(surface_points)

    for triangle in surface_triangles:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]
        p_s = [[p1, p2, p3]]
        colors = [get_color(max(*[p[0][2], p[1][2], p[2][2]])-min_height,
                            max_height-min_height, surface_base, change_range, inverse) for p in p_s]
        ax.add_collection3d(Poly3DCollection(p_s, facecolor=colors, alpha=0.9))

    # draw inner support first
    if inner_sup_points is not None and len(inner_sup_points)>0:
        # minius a little bit in height
        inner_sup_points = [[p[0], p[1], p[2]-1e-2] for p in inner_sup_points]
        # draw the supporting line
        # for p in inner_sup_points:
        #     x=[p[0],p[0]]
        #     y=[p[1],p[1]]
        #     z=[0,p[2]]
        #     ax.plot3D(x,y,z, color='red')
        support_triangles = get_triangles(inner_sup_points)
        for triangle in support_triangles:
            p1 = triangle[0]
            p2 = triangle[1]
            p3 = triangle[2]
            p_s = [[p1, p2, p3]]
            colors = [hsv_to_rgb(support_color, 1, 1) for _ in p_s]
            ax.add_collection3d(Poly3DCollection(p_s, facecolor=colors))

    ax.set_zlim3d(min_height, 1.25 * max_height)
    ax.set_title(title,  loc='center')
    # plt.legend()
    plt.show()

def draw_boundary_only(r: Callable, num:int=100):
    y_bottom= y_top = x_right = x_left = np.linspace(0,1,num)
    x_top = y_left = np.repeat(0, num)
    x_bottom = y_right = np.repeat(1, num)
    z_left = r(x_left, y_left)
    z_right = r(x_right, y_right)
    z_top = r(x_top, y_top)
    z_bottom = r(x_bottom, y_bottom)
    fig = plt.figure()
    axes3d = Axes3D(fig)
    axes3d.plot(x_left,y_left,z_left)
    axes3d.plot(x_right,y_right,z_right)
    axes3d.plot(x_top,y_top,z_top)
    axes3d.plot(x_bottom,y_bottom,z_bottom)
    plt.show()

class ObstacleGenerator:

    def __init__(self, n: int, base_val: float=0, min_val: float=0, max_val: float=2):
        self.n = n
        self.base_val = base_val
        self.min_val = min_val
        self.max_val = max_val
        self.obstacle = np.tile(float(base_val), (n, n))

    def add_point(self, x: int=None, y: int=None, min_val: float=None, max_val:float=None):
        if min_val is None: min_val = self.min_val
        if max_val is None: max_val = self.max_val
        if x is None: x = randint(0, self.n-2)
        if y is None: y = randint(0, self.n-2)
        self.obstacle[x][y] = uniform(min_val, max_val)

    def add_unit_rect(self, x: int=None, y: int=None, min_val: float=None, max_val:float=None):
        if min_val is None: min_val = self.min_val
        if max_val is None: max_val = self.max_val
        if x is None: x = randint(0, self.n-2)
        if y is None: y = randint(0, self.n-2)

        self.obstacle[x][y] = uniform(min_val, max_val)
        self.obstacle[x][y+1] = uniform(min_val, max_val)
        self.obstacle[x+1][y] = uniform(min_val, max_val)
        self.obstacle[x+1][y+1] = uniform(min_val, max_val)

    def add_unit_up_tri(self, x: int=None, y: int=None, min_val: float=None, max_val:float=None):
        if min_val is None: min_val = self.min_val
        if max_val is None: max_val = self.max_val
        if x is None: x = randint(0, self.n-2)
        if y is None: y = randint(0, self.n-2)

        self.obstacle[x][y] = uniform(min_val, max_val)
        self.obstacle[x+1][y] = uniform(min_val, max_val)
        self.obstacle[x+1][y+1] = uniform(min_val, max_val)

    def add_unit_down_tri(self, x: int=None, y: int=None, min_val: float=None, max_val:float=None):
        if min_val is None: min_val = self.min_val
        if max_val is None: max_val = self.max_val
        if x is None: x = randint(0, self.n-2)
        if y is None: y = randint(0, self.n-2)

        self.obstacle[x][y] = uniform(min_val, max_val)
        self.obstacle[x+1][y] = uniform(min_val, max_val)
        self.obstacle[x][y+1] = uniform(min_val, max_val)
