"""Small documented-API probe before enabling the new optional solver."""
import json
import cuopt
from cuopt.linear_programming import SolverMethod
from cuopt.linear_programming.problem import Problem, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

print("cuopt", cuopt.__version__, flush=True)
problem = Problem("gpu_probe")
x = problem.addVariable(lb=0, ub=3, name="x")
y = problem.addVariable(lb=0, ub=4, name="y")
problem.addConstraint(x+y <= 5)
problem.setObjective(2*x+y, sense=MAXIMIZE)
for method in (SolverMethod.Barrier, SolverMethod.PDLP):
    settings = SolverSettings()
    for key, value in dict(method=method, crossover=False, presolve=2,
                           log_to_console=False, time_limit=10.0).items():
        settings.set_parameter(key, value)
    settings.set_optimality_tolerance(1e-8)
    problem.solve(settings)
    print("method", method, "status", problem.Status, "x", x.Value, "y", y.Value,
          "obj", problem.ObjValue, flush=True)
    print("stats", problem.SolutionStats, flush=True)
    print("statvars", vars(problem.SolutionStats), flush=True)
    print("problem keys", list(vars(problem)), flush=True)
