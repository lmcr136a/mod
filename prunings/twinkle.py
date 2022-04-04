import numpy as np
import math
import torch
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

num_pruned_tolerate_coeff = 1.1

from sklearn.linear_model import Lasso





def channel_selection(inputs, module, tw_d, sparsity=0.5):
    num_channel = inputs.size(1)  # 채널 수
    num_pruned = int(math.ceil(num_channel * sparsity))  # 입력된 sparsity 에 맞춰 삭제되어야 하는 채널 수
    num_stayed = num_channel - num_pruned

    print('num_pruned', num_pruned)

    y = module(inputs)

    if module.bias is not None:  # bias.shape = [N]
        bias_size = [1] * y.dim()  # bias_size: [1, 1, 1, 1]
        bias_size[1] = -1  # [1, -1, 1, 1]
        bias = module.bias.view(bias_size)  # bias.view([1, -1, 1, 1] = [1, N, 1, 1])
        y -= bias  # output feature 에서 bias 만큼을 빼줌 (y - b)
    else:
        bias = 0.
    y = y.view(-1).data.cpu().numpy()  # flatten all of outputs
    y_channel_spread = []
    for i in range(num_channel):
        x_channel_i = torch.zeros_like(inputs)
        x_channel_i[:, i, ...] = inputs[:, i, ...]
        y_channel_i = module(x_channel_i) - bias
        y_channel_spread.append(y_channel_i.data.view(-1, 1))
    y_channel_spread = torch.cat(y_channel_spread, dim=1).cpu()

    alpha = 1e-7
    print("start twinkle_lasso")


    solver = Lasso(twinkle=True, tw=tw_d, alpha=alpha, random_state=0)
    # 원하는 수의 채널이 삭제될 때까지 alpha 값을 조금씩 늘려나감
    alpha_l, alpha_r = 0, alpha
    num_pruned_try = 0
    while num_pruned_try < num_pruned:
        alpha_r *= 2
        solver.alpha = alpha_r
        # solver.fit(selected_y_channel_spread, new_output)
        solver.fit(y_channel_spread,y)
        num_pruned_try = sum(solver.coef_ == 0)

    # 충분하게 pruning 되는 alpha 를 찾으면, 이후 alpha 값의 좌우를 좁혀 나가면서 좀 더 정확한 alpha 값을 찾음
    num_pruned_max = int(num_pruned)
    while True:
        alpha = (alpha_l + alpha_r) / 2
        solver.alpha = alpha
        # solver.fit(selected_y_channel_spread, new_output)
        solver.fit(y_channel_spread,y)
        num_pruned_try = sum(solver.coef_ == 0)

        if num_pruned_try > num_pruned_max:
            alpha_r = alpha
        elif num_pruned_try < num_pruned:
            alpha_l = alpha
        else:
            break
    # 마지막으로, lasso coeff를 index로 변환
    indices_stayed = np.where(solver.coef_ != 0)[0].tolist()
    indices_pruned = np.where(solver.coef_ == 0)[0].tolist()

    # solver = Twinkle_Lasso(
    #     # loss_function=mean_absolute_percentage_error,   ##########mean_squared_error##############################
    #     loss_function=eval(loss_function),
    #     alpha=alpha,
    #     tw_d=tw_d,
    # )
    # alpha_l, alpha_r = 0, alpha
    # num_pruned_try = 0
    # while num_pruned_try < num_pruned:
    #     alpha_r *= 2
    #     solver.alpha = alpha_r
    #     print(solver.beta)
    #     solver.fit(y_channel_spread,y)
    #     num_pruned_try = sum(solver.beta == 0)
    #     print(num_pruned_try)  ############################

    # # 충분하게 pruning 되는 alpha 를 찾으면, 이후 alpha 값의 좌우를 좁혀 나가면서 좀 더 정확한 alpha 값을 찾음
    # num_pruned_max = int(num_pruned)
    # while True:
    #     alpha = (alpha_l + alpha_r) / 2
    #     solver.alpha = alpha
    #     solver.fit(y_channel_spread,y)
    #     num_pruned_try = sum(solver.beta == 0)

    #     if num_pruned_try > num_pruned_max:
    #         alpha_r = alpha
    #     elif num_pruned_try < num_pruned:
    #         alpha_l = alpha
    #     else:
    #         break

    # # 마지막으로, lasso coeff를 index로 변환
    # indices_stayed = np.where(solver.beta != 0)[0].tolist()
    # indices_pruned = np.where(solver.beta == 0)[0].tolist()

    inputs = inputs.cuda()
    module = module.cuda()

    return indices_stayed, indices_pruned  # 선택된 채널의 인덱스를 리턴


def module_surgery(module ,attached_module,next_module, indices_stayed):
    """
    선택된 less important 필터/채널을 프루닝합니다.
    :param module: torch.nn.module, module of the Conv layer (be pruned for filters)
    :param attached_module: torch.nn.module, series of modules following the this layer (like BN)
    :param next_module: torch.nn.module, module of the next layer (be pruned for channels)
    :param indices_stayed: list of int, indices of channels and corresponding filters to be pruned
    :return:
        void
    """

    num_channels_stayed = len(indices_stayed)

    if module is not None:
        if isinstance(module, torch.nn.Conv2d):
            module.out_channels = num_channels_stayed
        elif isinstance(module, torch.nn.Linear):
            module.out_features = num_channels_stayed
        else:
            raise NotImplementedError

        # redesign module structure (delete filters)
        new_weight = module.weight[indices_stayed, ...].clone()
        del module.weight
        module.weight = torch.nn.Parameter(new_weight)
        if module.bias is not None:
            new_bias = module.bias[indices_stayed, ...].clone()
            del module.bias
            module.bias = torch.nn.Parameter(new_bias)

    # redesign BN module
    if attached_module is not None:
        if isinstance(attached_module, torch.nn.modules.BatchNorm2d):
            attached_module.num_features = num_channels_stayed
            running_mean = attached_module.running_mean[indices_stayed, ...].clone()
            running_var = attached_module.running_var[indices_stayed, ...].clone()
            new_weight = attached_module.weight[indices_stayed, ...].clone()
            new_bias = attached_module.bias[indices_stayed, ...].clone()
            del attached_module.running_mean, attached_module.running_var, attached_module.weight, attached_module.bias
            attached_module.running_mean, attached_module.running_var = running_mean, running_var
            attached_module.weight, attached_module.bias = torch.nn.Parameter(new_weight), torch.nn.Parameter(new_bias)

    # redesign next module structure (modify input channels)
    if next_module is not None:
        if isinstance(next_module, torch.nn.Conv2d):
            next_module.in_channels = num_channels_stayed
        elif isinstance(next_module, torch.nn.Linear):
            next_module.in_features = num_channels_stayed
        new_weight = next_module.weight[:, indices_stayed, ...].clone()
        del next_module.weight
        next_module.weight = torch.nn.Parameter(new_weight)


def weight_reconstruction(module, inputs, outputs, use_gpu=False):
    """
    이전 레이어의 프루닝이 수행되고 나면, 현재 레이어의 pruned output 을 이용하여 다음 레이어의 weight 를 조정합니다
    프루닝 된 레이어의 output 을 X로, 다음 레이어의 output 값(이 때 original model's output 값)을 Y로 두고 least square 를 풀게 됩니다.
    이렇게 구해진 parameter 를 다음 레이어의 weight 로 삼게 됩니다.
    reconstruct the weight of the next layer to the one being pruned
    :param module: torch.nn.module, module of the this layer
    :param inputs: torch.Tensor, new input feature map of the this layer
    :param outputs: torch.Tensor, original output feature map of the this layer
    :param use_gpu: bool, whether done in gpu
    :return:
        void
    """
    if use_gpu:
        inputs = inputs.cuda()
        module = module.cuda()

    if module.bias is not None:
        bias_size = [1] * outputs.dim()
        bias_size[1] = -1
        outputs -= module.bias.view(bias_size)  # output feature 에서 bias 만큼을 빼줌 (y - b)
    if isinstance(module, torch.nn.Conv2d):
        unfold = torch.nn.Unfold(kernel_size=module.kernel_size, dilation=module.dilation,
                                 padding=module.padding, stride=module.stride)
        if use_gpu:
            unfold = unfold.cuda()
        unfold.eval()
        x = unfold(inputs)  # 하나의 패치(reception field)를 열로 하는 3차원배열로 펼침 (N * KKC * L(number of fields))
        x = x.transpose(1, 2)  # 텐서를 transpose (N * KKC * L) -> (N * L * KKC)
        num_fields = x.size(0) * x.size(1)
        x = x.reshape(num_fields, -1)  # x: (NL * KKC)
        y = outputs.view(outputs.size(0), outputs.size(1), -1)  # feature map 하나를 행으로 하는 배열로 펼침 (N * C * WH)
        y = y.transpose(1, 2)  # 텐서를 transpose (N * C * HW) -> (N * HW * C), L == HW
        y = y.reshape(-1, y.size(2))  # y: (NHW * C),  (NHW) == (NL)

        if x.size(0) < x.size(1) or use_gpu is False:
            x, y = x.cpu(), y.cpu()
    #         x,y = x.cpu(), y.cpu()

    param, residuals, rank, s = np.linalg.lstsq(x.detach().cpu().numpy(),y.detach().cpu().numpy(),rcond=-1)
    # param, _ = torch.lstsq(y, x)
    if use_gpu:
        # param = param.cuda()
        param = torch.from_numpy(param).cuda()

    param = param[0:x.size(1), :].clone().t().contiguous().view(y.size(1), -1)
    if isinstance(module, torch.nn.Conv2d):
        param = param.view(module.out_channels, module.in_channels, *module.kernel_size)
    del module.weight
    module.weight = torch.nn.Parameter(param)



if __name__ == "__main__":
        
    # Generate predictors
    X_raw = np.random.random(100*9)
    X_raw = np.reshape(X_raw, (100, 9))

    # Standardize the predictors
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # Add an intercept column to the model.
    X = np.abs(np.concatenate((np.ones((X.shape[0],1)), X), axis=1))

    # Define my "true" beta coefficients
    beta = np.array([2,6,7,3,5,7,1,2,2,8])

    # Y = Xb
    Y_true = np.matmul(X,beta)

    # Observed data with noise
    Y = Y_true*np.exp(np.random.normal(loc=0.0, scale=0.2, size=100))


    l2_mape_model = Twinkle_Lasso(
        loss_function=mean_absolute_percentage_error, alpha=0.00012
    )
    l2_mape_model.fit()
    print(l2_mape_model.beta)