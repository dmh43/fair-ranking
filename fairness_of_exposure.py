import cvxpy as cp
import numpy as np

u = np.array((0.81, 0.80, 0.79, 0.78, 0.77, 0.76))
# u = np.array((0.81, 0.80, 0.79, 0.78, 0.77, 0.0))
v = np.array([1.0/(np.log(2 + i)) for i, _ in enumerate(u)])
P = cp.Variable((len(u), len(u)))
objective = cp.Maximize(cp.matmul(cp.matmul(u, P), v))
constraints = [cp.matmul(np.ones((1,len(u))), P) == np.ones((1,len(u))),
               cp.matmul(P, np.ones((len(u),))) == np.ones((len(u),)),
               0 <= P, P <= 1,
               cp.matmul(cp.matmul(np.array([1/u[:3].sum(), 1/u[:3].sum(), 1/u[:3].sum(), -1/u[3:].sum(), -1/u[3:].sum(), -1/u[3:].sum()]) / 3, P), v) == 0]
               # ]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
# result = prob.solve(verbose=True, max_iters=1000)
result = prob.solve(verbose=True, solver=cp.SCS)
# The optimal value for x is stored in `x.value`.
print(P.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)
