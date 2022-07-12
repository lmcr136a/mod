import numpy as np
import yaml
from utils.utils import configuration, show_test_acc, write_result

# TARGET_CLASSES = [[30, 78, 98, 17, 93], [63, 95,  2, 73, 14]]

# for i, tc in enumerate(TARGET_CLASSES):
for num in range(1, 6):
    for k in [5, 10, 15, 20, 25, 30]:
        path = f'configs/lasso4_{num}_{k}.yml'
        with open(path, encoding="utf-8") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)   

        cfg["network"]['pruning']["process_num"] = 4
        with open(path, "w") as f:
            yaml.dump(cfg, f)

for num in range(2, 3):
    for k in [25, 30]:
        path = f'configs/lasso2_{num}_{k}.yml'
        with open(path, encoding="utf-8") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)   

        cfg["network"]['pruning']["process_num"] = 4
        with open(path, "w") as f:
            yaml.dump(cfg, f)

for num in range(2, 3):
    for k in [25]:
        path = f'configs/lasso3_{num}_{k}.yml'
        with open(path, encoding="utf-8") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)   

        cfg["network"]['pruning']["process_num"] = 4
        with open(path, "w") as f:
            yaml.dump(cfg, f)
# string = ''
# for num in range(1, 7):
#     for i in [4,5]:
#         for k in [5, 10, 15, 20, 25, 30]:
#             string += f'lasso{i}_{num}_{k}.yml '

#     print(string)
#     string = ''
# for i in [3]:
#     for num in [2, 6]:
#         for k in [5, 10, 15, 20, 25, 30]:
#             string += f'lasso{i}_{num}_{k}.yml '

# print(string)
# string = ''