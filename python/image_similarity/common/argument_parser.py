import argparse

algorithms = [ 'mse', 'rmse', 'psnr', 'uqi', 'ssim', 'ergas', 'scc', 'rase', 'sam', 'msssim', 'vifp' ]

parser = argparse.ArgumentParser(description="Image Similarity", add_help=False)

argument_prefix = '--'

def init_args():
    for algorithm in algorithms :
        parser.add_argument(argument_prefix + algorithm, action="store_true", default=False, help="Calculates image similarity using the " + algorithm + " algorithm")    
    parser.add_argument("--all", action="store_true", help="Calculates image similarity using the all the algorithms")    
    parser.add_argument("--img_a", nargs=1, action="store", help="Path of first image")
    parser.add_argument("--img_b", nargs=1, action="store", help="Path of second image")
    parser.add_argument("--verbose", action="store_true", default=False, help="If passed, prints logs on every algorithm call")
    parser.add_argument("-h", "--help", action="help", help="Show this message and leaves.")


def parse_args ():
    init_args()
    return parser.parse_args()


def interpret_args ():
    args = parse_args()
    filtered_algorithms = [ algorithm for algorithm in algorithms if args.__dict__[algorithm] == True ]
    if len(filtered_algorithms) == 0 and args.all:
        filtered_algorithms = [ algorithm for algorithm in algorithms ]
    if len(args.img_a) > 0 and len(args.img_b) > 0:
        files = [ args.img_a[0], args.img_b[0] ]
    return filtered_algorithms, files, args.verbose
