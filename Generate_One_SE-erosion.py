# This program generates input-output images where output image is the result
# of performing 1 dilation transformation using one of the structuring elements # SE 1-8.
# Run it using: python Generate_One_SE.py
# It writes the images to DataSet/One_SE/Task001.json to Task008.json
# (one JSON file per SE).
# Each JSON file has a list of samples, each sample is 1 object in the format: 
#       {"input":  [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                   .... # 15 x 15 pixels
#                   [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]],
#        "output": [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                   .... # 15 x 15 pixels
#                   [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]],
#       }


import os
import numpy as np
import json
import pdb
from matplotlib import pyplot as plt

# from skimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import binary_dilation, binary_erosion, binary_hit_or_miss
import random

from ListSelEm import *
from Utils import Process, Change_Colour

import re


def generate_inp_out_catA_Simple(list_se_idx, **param):
    """
    """
    base_img = np.zeros((param['img_size'], param['img_size']), dtype=np.int32)
    sz = np.random.randint(3, 6)
    idx1 = np.random.randint(0, param['img_size'], size=sz)
    idx2 = np.random.randint(0, param['img_size'], size=sz)
    base_img[idx1, idx2] = 1

    # Select a random SE to dilate the base image
    # This way the inputs would have some structure but still remain random.
    for _ in range(2):
        idx = np.random.randint(0, 8)
        base_img = binary_erosion(base_img, list_se_3x3[idx])

    inp_img = np.array(base_img, copy=True)
    out_img = np.array(base_img, copy=True)

    # for idx in range(len(list_se_idx)):
       # out_img = binary_dilation(out_img, list_se_3x3[list_se_idx[idx]])

    # Arjun: no erosion
    for idx in range(len(list_se_idx)):
        out_img = binary_erosion(out_img, list_se_3x3[list_se_idx[idx]])

    return inp_img, out_img


def generate_images_one_SE(se_label, **param):
    """
    """
    # Arjun: only 1 SE
    list_se_idx = [se_label]
    data = []
    k = 0
    while k < param['no_examples_per_task']:
        inp_img, out_img = generate_inp_out_catA_Simple(list_se_idx, **param)

        # Check if both input and output images are non-trivial
        FLAG = False
        if np.all(inp_img*1 == 1) or np.all(inp_img*1 == 0):
            FLAG = True
        elif np.all(out_img*1 == 1) or np.all(out_img*1 == 0):
            FLAG = True

        if FLAG:
            k -= 1  # this image-pair is not counted
        else:
            # If not trivial proceed.
            data.append((inp_img, out_img))
        k += 1
        if k > 0 and k % 1000 == 0:
            print(f"Generated {k} image pairs")

    return data, list_se_idx


def write_dict_json_CatA_Simple(data, fname):
    """
    """
    dict_data = []
    for (inp, out) in data:
        inp = [[int(y) for y in x] for x in inp]
        out = [[int(y) for y in x] for x in out]
        dict_data.append({"input": inp, "output": out})

    with open(fname, "w") as f:
        data_str = json.dumps(dict_data)
        # Insert \n at the right places so image pixels appear as nice squares
        data_str = re.sub("\[\[", "\n[[", data_str)
        data_str = re.sub("\],", "],\n", data_str)
        data_str = re.sub("\},", "},\n", data_str)
        f.write(data_str)


def write_solution_CatA_Simple(list_se_idx, fname):
    """
    """
    with open(fname, 'w') as f:
        # for idx in list_se_idx:
           # f.write("Dilation SE{}\n".format(idx+1))
        # Arjun: no erosion
        for idx in list_se_idx:
            f.write("Erosion SE{}\n".format(idx+1))


def generate_tasks_one_SE(seed, **param):
    """
    """
    np.random.seed(seed)
    os.makedirs("./Dataset/One_SE-erosion", exist_ok=True)
    for se_label in range(8):
        data, list_se_idx = generate_images_one_SE(se_label, **param)
        fname = './Dataset/One_SE-erosion/Task{:03d}.json'.format(se_label)
        write_dict_json_CatA_Simple(data, fname)

        fname = './Dataset/One_SE-erosion/Task{:03d}_soln.txt'.format(se_label)
        write_solution_CatA_Simple(list_se_idx, fname)
        print(f"Saved data for task {se_label}")


if __name__ == "__main__":
    param = {}
    param['img_size'] = 15
    #param['se_size'] = 5  # Size of the structuring element.  Always 3x3
    #param['seq_length'] = 4  # Number of primitives would be 2*param['seq_length']
    param['no_examples_per_task'] = 10000
    #param['no_colors'] = 3

    generate_tasks_one_SE(32, **param)
