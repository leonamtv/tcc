from core.algorithms.mse import mse
from core.algorithms.rmse import rmse
from core.algorithms.psnr import psnr
from core.algorithms.uqi import uqi
from core.algorithms.ssim import ssim
from core.algorithms.ergas import ergas
from core.algorithms.scc import scc
from core.algorithms.rase import rase
from core.algorithms.sam import sam
from core.algorithms.msssim import msssim
from core.algorithms.vifp import vifp
from numpy import float64

strategy_map = {
    'mse' : mse,
    'rmse' : rmse,
    'psnr' : psnr,
    'uqi' : uqi,
    'ssim' : ssim,
    'ergas' : ergas,
    'scc' : scc,
    'rase' : rase,
    'sam' : sam,
    'msssim' : msssim,
    'vifp' : vifp
}


def run_algorithm(algorithm_name: str, img_path_1: str, img_path_2: str, verbose: bool = False) -> float64:
    return strategy_map[algorithm_name](img_path_1, img_path_2, verbose)


def run_algorithms(algorithms: list, img_path_1: str, img_path_2: str, verbose: bool = False) :
    result_map = {}
    for algorithm in algorithms:
        result_map[algorithm] = str(run_algorithm(algorithm, img_path_1, img_path_2, verbose))
    return result_map
