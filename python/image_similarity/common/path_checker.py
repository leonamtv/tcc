from os.path import exists, isfile

def check_paths(paths: list):
    for path in paths :
        check_path(path)

def check_path(path: str) :
    if not exists(path) or not isfile(path) :
        raise Exception("Path {0} does not exist or isn't a file".format(path))