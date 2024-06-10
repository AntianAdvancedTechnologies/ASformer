import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import time
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="cleanjerk")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='model')
parser.add_argument('--result_dir', default='result')

args = parser.parse_args()
 
num_epochs = 100

lr = 0.0001
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 2
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#     sample_rate = 2

# # To prevent over-fitting for GTEA. Early stopping & large dropout rate
# if args.dataset == "gtea":
#     channel_mask_rate = 0.5
    
# if args.dataset == 'breakfast':
#     lr = 0.0001

vid_list_file = "ASformer/data/splits/train.split1.bundle"
vid_list_file_tst = "ASformer/data/splits/test.split1.bundle"
features_path = "ASformer/data/features/"
gt_path = "ASformer/data/groundTruth/"
 
# mapping_file = "snatch\\mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
# file_ptr = open(mapping_file, 'r')
# actions = file_ptr.read().split('\n')[:-1]
# file_ptr.close()

#Snatch
# actions = ['Setup', 'first_pull', 'transition', 'explosion', 'final_extension', 'turnover', 'amortization', 'recovery', 'snatch_done', 'No_action']

# actions_dict =  {
#     'setup': 0,
#     'first_pull': 1,
#     'transition': 2,
#     'explosion': 3,
#     'final_extension': 4,
#     'turnover': 5,
#     'amortization': 6,
#     'recovery': 7,
#     'snatch_done': 8,
#     'No_action': 9
# }

#CleanJerk
actions = ['first_pull', 'transition', 'explosion', 'final_extension', 'turnover', 'amortization', 'recovery', 'clean_done', 'dip', 'drive', 'split_under', 'jerk_done', 'No_action']

actions_dict = {
    0: 'first_pull',
    1: 'transition',
    2: 'explosion',
    3: 'final_extension',
    4: 'turnover',
    5: 'amortization',
    6: 'recovery',
    7: 'clean_done',
    8: 'dip',
    9: 'drive',
    10: 'split_under',
    11: 'jerk_done',
    12: 'No_action'
}



for a in actions:
    actions_dict[a] = actions.index(a)

# Create index2label dictionary
index2label = {v: k for k, v in actions_dict.items()}


# actions_dict['No_action'] = 9
# actions_dict
num_classes = len(actions_dict)
# num_classes


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)

if args.action == "train":
    start_time = time.time()
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)
    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total time taken
    total_minutes = total_time / 60
    print(f"Total time taken for training: {int(total_minutes)} minutes")

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

# python Asformer/main.py --action=train --dataset=snatch --split=1 
