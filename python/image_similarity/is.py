from common.argument_parser import interpret_args
from core.algorithm_strategy import run_algorithms

algorithms, files, verbose = interpret_args()

if len(files) > 1:
    print(run_algorithms(algorithms, files[0], files[1], verbose))

