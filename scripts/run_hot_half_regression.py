"""Repeat the prior numerical regression plus new experimental tests."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import pytest
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results/pf_ipm_half_target_final_regression_20260907.xml'
paths={c.attrib['classname'].replace('.','/')+'.py' for c in ET.parse(source).iter('testcase')}
paths.update('tests/'+name for name in ('test_gpu_dual_schur.py','test_gpu_centrality_corrector.py',
    'test_gpu_primal_dual_step.py','test_gpu_dr_feasibility.py','test_gpu_shared_newton.py',
    'test_gpu_feasibility_newton.py','test_gpu_ray_reoptimization.py',
    'test_gpu_block_affine_feasibility.py','test_gpu_primal_extrapolation.py',
    'test_supervised_graph_lp.py','test_graph_primal_restart.py',
    'test_graph_temporal_lp.py','test_graph_ipm_bridge.py','test_graph_learning_curve.py',
    'test_graph_collection_coordinator.py','test_offline_teacher_recovery.py'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('Refusing to overwrite prior regression evidence')
raise SystemExit(pytest.main(['-q',*sorted(paths),'--junitxml='+str(args.output)]))
