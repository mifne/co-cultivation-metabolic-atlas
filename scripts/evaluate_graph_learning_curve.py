"""Summarize measured scaling, never use correlated LP steps as repetitions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.learning_curve_evaluation import evaluate_curve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resamples',type=int,default=2000)
    a=p.parse_args()
    result=evaluate_curve(json.loads(a.input.read_text()),resamples=a.resamples)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['decision']))


if __name__=='__main__':main()
