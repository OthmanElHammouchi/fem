# benchmark
import sympy
import numpy as np
import matplotlib.pyplot as plt

name = "Othman El Hammouchi"
vec = np.round(np.cumsum(list(map(ord, name))) / 20 + np.arange(1, len(name) + 1))
k_vals = 5 + np.arange(1, len(vec)) ** 1.2
q_idxs = list(map(int, [0, np.ceil(len(vec) / 3) - 1, np.ceil(len(vec) * 2 / 3) - 1]))
q1, q2, q3 = -2.3, 1.4, -3.9
q_vals = np.zeros(len(k_vals))
q_vals[q_idxs] = [q1, q2, q3]
a = vec[0]
b = vec[-1]
Ta = 100
bc = []
bc.append((vec[-1], Ta))

N = int(1e2)
h = 1 / N
n_elems = int((b - a) / h)
n_nodes = n_elems + 1

k, q = sympy.symbols("k q", cls=sympy.Function)
x_sym = sympy.symbols("x")

q_expr_conds = []
k_expr_conds = []
for i in range(len(vec) - 1):
    l_point = vec[i]
    r_point = vec[i + 1]
    k_expr_cond = (k_vals[i], (x_sym < r_point) & (x_sym >= l_point))
    q_expr_cond = (q_vals[i], (x_sym < r_point) & (x_sym >= l_point))
    k_expr_conds.append(k_expr_cond)
    q_expr_conds.append(q_expr_cond)

k_expr_conds.append((k_vals[-1], (sympy.Eq(x_sym, vec[-1]))))
q_expr_conds.append((q_vals[-1], (sympy.Eq(x_sym, vec[-1]))))

k = sympy.Piecewise(*k_expr_conds)
q = sympy.Piecewise(*q_expr_conds)

k = sympy.lambdify(x_sym, k)
q = sympy.lambdify(x_sym, q)

from fenics import *

mesh = IntervalMesh(n_elems, vec[0], vec[-1])
V = FunctionSpace(mesh, "P", 3)

def boundary_L(x, on_boundary):
    r = on_boundary and np.abs(x - vec[0]) < 1e-4
    return r

left_val = Constant(Ta)
bc = DirichletBC(V, left_val, boundary_L)

T = TrialFunction(V)
v = TestFunction(V)

class f(UserExpression):
    def eval(self, value, x):
      value[0] = q(x[0]) / k(x[0])

f = f(element = V.ufl_element())

a = dot(grad(T), grad(v)) * dx
L = f * v * dx

T = Function(V)
solve(a == L, T, bc)

plot(T)
plt.show()