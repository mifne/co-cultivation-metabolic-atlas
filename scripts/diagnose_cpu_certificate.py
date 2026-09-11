"""Isolated CPU-only diagnostics for a saved original LP certificate failure."""
import argparse
import json
from pathlib import Path
import sys
import time

import highspy
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cpu_repeated_lp import _certificate, _NUMERICAL_OPTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.problem, allow_pickle=False) as data:
        a = csr_matrix((data['a_data'], data['a_indices'], data['a_indptr']), shape=tuple(data['a_shape']))
        rhs, lower, upper, c = (data[key].copy() for key in ('rhs', 'lower', 'upper', 'c'))
        neq = int(data['neq'])
    rows, columns = a.shape

    def make(tolerance=None, presolve='on'):
        solver = highspy.Highs()
        options = dict(threads=1, parallel='off', solver='simplex', simplex_strategy=1,
                       output_flag=False, presolve=presolve)
        if tolerance is not None:
            options.update({name:tolerance for name in _NUMERICAL_OPTIONS[:-1]})
        for key, value in options.items():
            assert solver.setOptionValue(key, value) == highspy.HighsStatus.kOk
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = columns, rows
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lower, upper
        lp.row_lower_, lp.row_upper_ = np.r_[rhs[:neq], np.full(rows-neq, -np.inf)], rhs
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = a.indptr, a.indices, a.data
        assert solver.passModel(lp) != highspy.HighsStatus.kError
        return solver

    def run(solver, label):
        started = time.perf_counter()
        status = solver.run()
        elapsed = time.perf_counter()-started
        solution, info = solver.getSolution(), solver.getInfo()
        x, y = np.asarray(solution.col_value), np.asarray(solution.row_dual)
        reduced, activity = c-a.T@y, a@x
        target = np.where(reduced >= 0., lower, upper)
        bound_gap = np.abs(reduced*(x-np.where(np.isfinite(target), target, x)))
        row_gap = np.abs(y[neq:]*(rhs-activity)[neq:])
        result = dict(label=label, run_status=str(status), model_status=str(solver.getModelStatus()),
            seconds=elapsed, simplex_iterations=info.simplex_iteration_count,
            value_valid=solution.value_valid, dual_valid=solution.dual_valid,
            effective_options={key:getattr(solver.getOptions(),key) for key in _NUMERICAL_OPTIONS},
            objective=float(c@x), bound_gap=float(bound_gap.sum()), row_gap=float(row_gap.sum()),
            **_certificate(a, rhs, lower, upper, c, neq, solution),
            highs_info={key:getattr(info,key) for key in ('max_primal_infeasibility',
                'max_dual_infeasibility', 'max_primal_residual_error', 'max_dual_residual_error',
                'max_complementarity_violation', 'primal_dual_objective_error')})
        print(json.dumps(result), flush=True)
        return result

    results = []
    solver = make()
    results.append(run(solver, 'cold_default_presolve'))
    for tolerance in (1e-9, 1e-10):
        basis = solver.getBasis()
        for name in _NUMERICAL_OPTIONS[:-1]:
            assert solver.setOptionValue(name, tolerance) == highspy.HighsStatus.kOk
        assert solver.clearSolver() == highspy.HighsStatus.kOk
        assert solver.setBasis(basis) == highspy.HighsStatus.kOk
        results.append(run(solver, f'refactor_previous_basis_{tolerance:g}'))
        results.append(run(make(tolerance, 'off'), f'cold_no_presolve_{tolerance:g}'))
    result = dict(source=str(args.problem), highs_version=highspy.Highs().version(),
        scope='Isolated original LP; preceding trajectory basis was not saved, so it is not reproduced.',
        shape=list(a.shape), nonzeros=a.nnz, minimum_nonzero=float(np.abs(a.data).min()),
        small_nonzero_count=int(np.count_nonzero(np.abs(a.data) <= 1e-9)), attempts=results)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
