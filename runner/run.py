import argparse
import sys
import importlib
from logger import log

# based off of sample factory's runner
# https://github.com/alex-petrenko/sample-factory


def runner_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, type=str, help='Name of the python module that describes the run, e.g. pymap_elites.runner.runs.bipedal_walker_runner')
    parser.add_argument('--runner', default='processes', choices=['processes', 'slurm'], help='Runner backend, use OS multiprocessing by default')
    return parser


def parse_args():
    args = runner_argparse().parse_args(sys.argv[1:])
    return args


def main():
    args = parse_args()
    try:
        # assuming we're given the full name of the module
        run_module = importlib.import_module(f'{args.run}')
    except ImportError:
        log.error('Could not import the run module')
        return 0


