import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import array_to_latex as a2l
from helpers import *

plot_dir = "plots"
res_dir = "results"
mpl.rcParams["text.usetex"] = True

name = "Othman El Hammouchi"
x_part = np.round(np.cumsum(list(map(ord, name))) / 20 + np.arange(1, len(name) + 1))
k_vals = 5 + np.arange(1, len(x_part)) ** 1.2
q_idxs = list(map(int, [0, np.ceil(len(x_part) / 3) - 1, np.ceil(len(x_part) * 2 / 3) - 1]))
q1, q2, q3 = -2.3, 10, -3.9
q_vals = np.zeros(len(k_vals))
q_vals[q_idxs] = [q1, q2, q3]
Ta = 100

N = int(1e2)
h = 1 / N
nodes, elems, bc = gen_mesh(h, x_part, k_vals, q_vals, Ta, randomise=True)
fem_sol = fem_solve(nodes, elems, h, bc)
exact_sol = exact_solve(x_part, k_vals, q_vals, bc)

elems_latex = a2l.to_ltx(elems[:10, :], arraytype="matrix", frmt="{:.2f}", print_out=False)
nodes_latex = a2l.to_ltx(nodes[:10], arraytype="matrix", frmt="{:.2f}", print_out=False)

with open(os.path.join(res_dir, "elems.tex"), "w") as elems_file:
    elems_file.write(elems_latex)
with open(os.path.join(res_dir, "nodes.tex"), "w") as nodes_file:
  nodes_file.write(nodes_latex)

np.savetxt(os.path.join(res_dir, "elems.txt"), elems[:10, :])
np.savetxt(os.path.join(res_dir, "nodes.txt"), nodes[:10])

x = np.linspace(x_part[0], x_part[-1], int(1e2))
y_fem = fem_sol(x)
y_exact = np.array(list(map(exact_sol, x)))
grad = np.gradient(y_fem)

width = 420 * 0.014
height = 630 * 0.014 / 2.5

fig, ax = plt.subplots(figsize=(width, height))
fig2, ax2 = plt.subplots(figsize=(width, height))

ax.plot(x, y_fem, "go", fillstyle='none', label="fem")
ax.plot(x, y_exact, color='orange', label="exact")
for i in range(len(q_idxs)):
    left = x_part[q_idxs[i]]
    right = x_part[q_idxs[i] + 1]
    ax.axvline(x=left, linestyle="--")
    ax.axvline(x=right, linestyle="--")
    ax.axvspan(left, right, alpha=0.5, color='skyblue')
    ax2.axvline(x=left, linestyle="--")
    ax2.axvline(x=right, linestyle="--")
    ax2.axvspan(left, right, alpha=0.5, color='skyblue')

    if q_vals[q_idxs[i]] != 0:
      x_an = (left + right) / 2 - 1
      y_an = sum(ax.get_ylim()) / 2
      ax.annotate('q = ' + str(q_vals[q_idxs[i]]), xy=(x_an, y_an), rotation = 90, fontsize = 12, fontstyle = 'italic')
      y_an = sum(ax2.get_ylim()) / 2
      ax2.annotate('q = ' + str(q_vals[q_idxs[i]]), xy=(x_an, y_an), rotation = 90, fontsize = 12, fontstyle = 'italic')

ax.legend()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T$", rotation=0)

ax2.plot(x, grad)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$\displaystyle \frac{dT}{dx}$", rotation=0)

fig.savefig(os.path.join(plot_dir, "sol.eps"), format="eps", dpi=200)
fig2.savefig(os.path.join(plot_dir, "grad.eps"), format="eps", dpi=200)

denoms = np.arange(1, 15)
n = len(denoms)
errors = np.empty(n)
for i in tqdm(range(n)):
  N = denoms[i]
  h = 1 / N
  nodes, elems, bc = gen_mesh(h, x_part, k_vals, q_vals, Ta, randomise=True)
  fem_sol = fem_solve(nodes, elems, h, bc, show_progress=False)
  exact_sol = exact_solve(x_part, k_vals, q_vals, bc)

  x = np.linspace(x_part[0], x_part[-1], int(1e6))
  y_fem = fem_sol(x)
  y_exact = np.array(list(map(exact_sol, x)))

  errors[i] = np.sqrt(((y_fem - y_exact)**2).mean())

fig3, ax3 = plt.subplots(figsize=(width, height))
ax3.loglog(1 / np.arange(1, n + 1), errors)
ax3.set_xlabel("$h$")
ax3.set_ylabel("$\Vert T_{\mathrm{FEM}} - T_{\mathrm{exact}} \Vert_2$")

fig3.savefig(os.path.join(plot_dir, "error.eps"), format="eps", dpi=200)