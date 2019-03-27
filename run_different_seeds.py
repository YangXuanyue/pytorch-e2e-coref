import time
import os
import random
from tqdm import tqdm

# for _ in tqdm(range(50)):
#     seed = random.randrange(1699780449)
#     os.system(f'CUDA_VISIBLE_DEVICES=5 python trainer.py -s {seed} -wd 1e-4 > /dev/null')

seed = random.randrange(1281967231)
os.system(f'CUDA_VISIBLE_DEVICES=5 python trainer.py -s {seed} > /dev/null')

for _ in tqdm(range(50)):
    seed = random.randrange(2281967231)
    os.system(f'CUDA_VISIBLE_DEVICES=5 python trainer.py -s {seed} > /dev/null')


    # with open('max_acc.txt') as max_acc_file:
    #     max_acc = max_acc_file.readline()
    #     print(max_acc)
