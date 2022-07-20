import numpy as np
import yaml
from utils.utils import configuration, show_test_acc, write_result
import os.path






print_string = ''
for ran_num in range(1, 2):
    for exp_num in [1, 4]:
        for k in [50, 60, 70, 80]:
            filename = f'lasso{ran_num}_{exp_num}_{k}.yml'
            path = f'configs/6set/{filename}'
            with open(path, encoding="utf-8") as fp:
                cfg = yaml.load(fp, Loader=yaml.FullLoader)   

            cfg["network"]['model'] = 'resnet56'
            cfg['data']['target_classes'] = [0,1,2]
            cfg["network"]['load_state'] = 'models/pretrained/resnet56_cifar10.pt'
            cfg['data']['for_test'] = cfg['data']['for_test'][:-1]
            cfg['data']['for_trainNval'] = cfg['data']['for_trainNval'][:-1]
            # if ran_num == 2:
            #     cfg['data']['target_classes'] = [40, 83, 25, 53, 7, 11, 93, 51, 26, 74]
            # if ran_num == 3:
            #     cfg['data']['target_classes'] = [40, 83, 25, 53, 7, 11, 93, 51, 26, 74, 30, 78, 98, 17, 93, 63, 95,  2, 73, 14]

            newname = f"rft/r56_cifar10_{exp_num}_{k}.yml"
            with open(f'configs/{newname}', "w") as f:
                yaml.dump(cfg, f)
            print_string+= f"{newname} "

print(print_string)


# # experiments 에서 가져와서 configs에 저장
# for ran_num in [2, 3]:
#     for exp_num in range(1, 7):
#         for k in [5, 10, 15, 20, 25, 30]:
#             filename = f'lasso{ran_num}_{exp_num}_{k}.yml'
#             path = f'configs/6set/{filename}'
#             if not os.path.isfile(path):
                
#                 with open(f'experiments/6set/lasso{ran_num}_r34_6set/lasso{ran_num}_{exp_num}_{k}/lasso_rand_{exp_num}_{k}.yml', encoding="utf-8") as fp:
#                     cfg = yaml.load(fp, Loader=yaml.FullLoader)   
#                 with open(path, "w") as f:
#                     yaml.dump(cfg, f)            
                    


# exp_path = 'experiments/6set/'
# big_dirs = [x[0] for x in os.walk(exp_path)]
# for bid_dir in big_dirs:
#     if os.path.isfile(bid_dir+"/total_indices_stayed_0.pt"):
#         os.rename(bid_dir+"/total_indices_stayed_0.pt", bid_dir+"/total_indices_stayed.pt")



# print_string = ''
# for tnum in [50, 100, 200, 300]:
#     # for k in [90, 80, 60, 40, 20]:
#     path = f'configs/pns/vgg_{tnum}.yml'
#     with open(path, encoding="utf-8") as fp:
#         cfg = yaml.load(fp, Loader=yaml.FullLoader)  

#     # cfg["network"]['load_state'] = 'from pytorch'
#     cfg['data']['target_classes'] = list(range(tnum))

#     with open( f'configs/pns/vgg_{tnum}.yml', "w") as f:
#         yaml.dump(cfg, f)

#     # print_string+= f"pns/vgg_{tnum}_{k}.yml "
        
# # print(print_string)