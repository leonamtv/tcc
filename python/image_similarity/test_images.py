import glob
import os
from config import paths
from core.algorithm_strategy import run_algorithms
from common.argument_parser import algorithms
from PIL import Image

results = {}

def check_size(img_a, img_b):
    im_a, im_b = Image.open(img_a), Image.open(img_b)
    width_a, height_a = im_a.size
    width_b, height_b = im_b.size
    if width_a < width_b and height_a < height_b :
        size = (width_a, height_a)
        im_b_r = im_b.resize(size, Image.LANCZOS)
        im_b_r.save(img_b, "PNG")
    if width_b < width_a and height_b < height_a :
        size = (width_b, height_b)
        im_a_r = im_a.resize(size, Image.LANCZOS)
        im_a_r.save(img_a, "PNG")


def get_equivalent(file_path, list_of_files) :
    file_name = os.path.basename(file_path)
    for file in list_of_files :
        if  os.path.basename(file) == file_name :
            return file
    return None


def run():
    for path in paths:
        name, og_p, at_p, ext = path['name'], path['original_files'], path['after_training_files'], path['extension']
        # print(name, og_p, at_p, ext)
        og_f, at_f = glob.glob(og_p + "*" + ext), glob.glob(at_p + "*" + ext)
        results[name] = {}
        for file_path in at_f:
            results[name][os.path.basename(file_path)] = {}
            equivalent_file_path = get_equivalent(file_path, og_f)
            # print(file_path, equivalent_file_path)
            check_size(equivalent_file_path, file_path)
            if equivalent_file_path is not None:
                results[name][os.path.basename(file_path)] = {
                    'original' : equivalent_file_path,
                    'after_training' : file_path,
                    'results' : run_algorithms(algorithms, equivalent_file_path, file_path, False)
                }

    print(results)

    
run()

        

def test_get_equivalent():
    print(get_equivalent("ASDASUDH/OSKFOSK123/AUSHDHAUDHA/654321.png", [ 
        "LKASJD9/102ujasd/ALÇKSD0K/654321.png",
        "LKASJD9/102ujasd/ALÇKSD0K/123456.png",
        "LKASJD9/102ujasd/ALÇKSD0K/86454a.png",
        "LKASJD9/102ujasd/ALÇKSD0K/549842.png",
        "LKASJD9/102ujasd/ALÇKSD0K/215648.png",
        "LKASJD9/102ujasd/ALÇKSD0K/124561.png"
    ]))