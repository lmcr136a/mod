import time
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
# import torch.nn.utils.prune as prune

from utils.metrics import AverageMeter, accuracy
from figure import plot_classes_preds
from torchsummaryX import summary

from utils.utils import show_test_acc, show_profile
from dataset import get_dataloader, get_test_dataloader
from models.model import get_network

from prunings.lasso import AssembleNetResNet
from prunings.twinkle_lasso import TwinkleAssembleNetResNet
from prunings.chip import calculate_ci, calculate_feature_map, prune_finetune_cifar
from prunings.hrank import hrank_main, rank_generation
from prunings.gal import get_gal_model, gal_main

# from prunings.nuc_l2_norm import calculate_nucl2, cal_fm, nucl2_prune


def run(cfg, writer):
    cfg_run = cfg["run"]
    dataloader, n_class = get_dataloader(cfg)
    get_test_dataloader(cfg, dataloader, get_only_targets=True)
    network = get_network(cfg["network"], n_class)

    device = is_cuda()
    np.random.seed(cfg_run["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(cfg_run["seed"])
        print(f"[ DEVICE  ] CUDA GPU available")

    network = network.to(device)
    criterion = get_loss(cfg_run)
    optimizer = get_optimizer(cfg_run, network)

    print(f"[LOSS FUNC] {criterion}  [OPTIMIZER] {optimizer}")

    if cfg["network"].get("load_state", None):
        print("\n====================== LOADING STATES... =====================")
        network.load_state_dict(torch.load(cfg["network"].get("load_state", None)),  strict=False)
    else:
        print("\n====================== TRAINING START! =====================")
        network = trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer)
        torch.save(network.state_dict(), writer.log_dir+"/best_model.pt")

    show_test_acc(test(dataloader, network, criterion, device))   ## CIFAR
    ################################################################

    summary(network, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
    show_profile(network)

    if cfg["network"].get("pruning", None):
        if cfg["network"]["pruning"].get("lasso", None) and not cfg["network"]["pruning"].get("iterative_lasso", None):
            lasso(cfg, dataloader, network, optimizer, criterion, n_class, writer.log_dir, device)
        elif cfg["network"]["pruning"].get("iterative_lasso", None):
            iterative_lasso(cfg, dataloader, network, optimizer, criterion, n_class, writer, device) ###
        elif cfg["network"]["pruning"].get("chip", None):
            network, optimizer, criterion = chip(network, cfg, dataloader, n_class, writer.log_dir)
        elif cfg["network"]["pruning"].get("hrank", None):
            network = hrank(cfg, network, device, dataloader, criterion, n_class, writer.log_dir)
        elif cfg["network"]["pruning"].get("gal", None):
            network = gal(dataloader, network, cfg, device, writer.log_dir, n_class)
        elif cfg["network"]["pruning"].get("twinkle", None):
            twinkle(cfg, dataloader, network, optimizer, criterion, n_class, writer.log_dir, device)

        summary(network, torch.zeros((2, 3, 32, 32)).to(torch.device("cuda")))
        show_profile(network)
        show_test_acc(test(dataloader, network, criterion, device))   ## CIFAR

        network = trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer)   ## INV
        show_test_acc(test(dataloader, network, criterion, device))   ## CIFAR
    


def iterative_lasso(cfg, dataloader, network, optimizer, criterion, n_class, writer, device):
    print("ITERATIVE LASSO START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    iter_num = cfg['run']['iter_num'] ## 6
    if not cfg['run'].get('hard_k', False):
        iter_k = cfg['network']['pruning']['k'] ** (1/iter_num)
        cfg['network']['pruning']['k'] = iter_k

    if cfg["network"]["pruning"]["all_classes"]:
        all_class_dataloader, _ = get_dataloader(cfg, get_only_targets=False)
    else:
        all_class_dataloader = None
        
    start = time.time()
    for i in range(iter_num):
        network = trainNval(dataloader, network, cfg['run'], criterion, optimizer, device, writer)
        lasso(cfg, dataloader, network, optimizer, criterion, n_class, writer.log_dir, \
            device, all_class_dataloader=all_class_dataloader, save_i=i)
        print(f'iter {i}.. {round((time.time()-start)//60)}min.. {round((time.time()-start)/3600, 1)}hour...')
    network = trainNval(dataloader, network, cfg['run'], criterion, optimizer, device, writer)


def lasso(cfg, dataloader, network, optimizer, criterion, n_class, log_dir, device, save_i=0, all_class_dataloader=None):
    # print("LASSO START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    agent = AssembleNetResNet(cfg["run"], dataloader, network, optimizer, criterion, n_class, device)   ## INV
    agent.init_graph(pretrained=False)
    if cfg["network"]["pruning"]["all_classes"]:
        if not all_class_dataloader:
            all_class_dataloader, _ = get_dataloader(cfg, get_only_targets=False)
        agent.data_loader = all_class_dataloader
    #     agent.compress(log_dir, method=cfg["network"]["pruning"].get("method", "lasso"), k=cfg["network"]["pruning"].get("k", 0.49))
    # else:
    agent.lasso_compress(cfg["network"]["pruning"], log_dir, save_i=save_i)
    

def twinkle(cfg, dataloader, network, optimizer, criterion, n_class, log_dir, device):
    print("twinkle START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    agent = TwinkleAssembleNetResNet(cfg["run"], dataloader, network, optimizer, criterion, n_class, device)   ## INV
    agent.init_graph(pretrained=False)
    if cfg["network"]["pruning"]["all_classes"]:
        all_class_dataloader, _ = get_dataloader(cfg, get_only_targets=False)
        agent.data_loader = all_class_dataloader
        agent.compress(log_dir, t_cfg=cfg["network"]["pruning"]["twinkle"])
    else:
        agent.lasso_compress(log_dir, t_cfg=cfg["network"]["pruning"]["twinkle"])





def chip(network, cfg, dataloader, n_class, log_dir):
    cfg_network = cfg["network"]
    print("CHIP START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    if cfg_network['pruning']["chip"].get("ci_path", None):
        ci_dir = cfg_network['pruning']["chip"]["ci_path"]
    else:
        if not cfg_network['pruning']["chip"].get("fm_path", None):
            if cfg_network["pruning"]["all_classes"]:
                dataloader, _ = get_dataloader(cfg, get_only_targets=False)
            calculate_feature_map(network, cfg_network["model"], dataloader, log_dir=log_dir)
            fm_path = log_dir
        else:
            fm_path = cfg_network['pruning']["chip"]["fm_path"]
        print("CAL CI START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")    
        ci_dir = calculate_ci(cfg_network["model"], log_dir, fm_path=fm_path)
    print("PRUNE START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")    
    network, optimizer, criterion = prune_finetune_cifar(cfg_network, cfg_network["load_state"], ci_dir, n_class)
    return network, optimizer, criterion

def hrank(cfg, network, device, dataloader, criterion, n_class, log_dir):
    print("HRank START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    cfg_network = cfg["network"]
    if cfg_network["pruning"]["all_classes"]:
        dataloader, _ = get_dataloader(cfg, get_only_targets=False)
    if not cfg_network["pruning"]["hrank"].get("rank_conv_path", None):
        print("CAL RANK START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
        rank_conv_path = rank_generation(cfg_network["model"], network, device, dataloader, criterion, log_dir)
        cfg_network["pruning"]["hrank"].update({"rank_conv_path": rank_conv_path})
    network = hrank_main(cfg_network, n_class, log_dir, device, dataloader)
    return network
    

def gal(dataloader, network, cfg, device, job_dir, n_class):
    print("GAL START mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    cfg_network = cfg["network"]
    if not cfg_network["pruning"]["gal"]["saved_path"]:
        if cfg_network["pruning"]["all_classes"]:
            dataloader, _ = get_dataloader(cfg, get_only_targets=False)
        gal_main(dataloader, network, cfg_network, device, job_dir, n_class)
    network = get_gal_model(cfg_network, device, n_class)
    return network


def trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer):
    """
    cfg_run에 담긴대로 training/validation 진행
    가장 높은 validation accuracy를 가진 네트워크를 출력

    여기서 data는 dictionary type이다.
    """
    best_acc = 0.0
    best_network = None
    TimeMeter = AverageMeter()
    start = time.time()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg_run["optimizer"].get("milestones", 10), gamma=cfg_run["optimizer"].get("gamma", 0.5))
    
    i = 0
    print_label = 0
    print_label_val = 0
    for epoch in range(cfg_run["epoch"]):
        BetchTimeMeter = AverageMeter()
        LossMeter = AverageMeter()
        Top1Meter = AverageMeter()
        end = time.time()

        network.train()

        for inputs, labels in dataloader.train_loader:
            if print_label == 0 :
                print_label = 1
                print(labels[:20])
            inputs = inputs.to(device)
            labels = labels.to(device)
            ## COMPUTE
            output = network(inputs)
            labels = labels.to(torch.long)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            ## METRICS
            prec1 = accuracy(output.data, labels)[0]
            LossMeter.update(loss.item(), inputs.size(0))
            Top1Meter.update(prec1.item(), inputs.size(0))

            BetchTimeMeter.update(time.time() - end)
            end = time.time()
        scheduler.step()
        if Top1Meter.avg >= best_acc:
            best_network = network
            best_acc = Top1Meter.avg

        if (epoch+1) % cfg_run["print_interval"] == 0:
            print(f'[Epoch {epoch+1}/{cfg_run["epoch"]}] [TRAIN] Loss {LossMeter.avg:.4f})\t Acc: {Top1Meter.avg:.3f})\t Time {BetchTimeMeter.avg:.3f})')

        if (epoch+1) % cfg_run["plot_interval"] == 0:
            writer.add_scalar('Training loss',
                            LossMeter.avg,
                            epoch)
        if (epoch+1) % cfg_run["figure_interval"] == 0:
            writer.add_figure('training prediction',
                            plot_classes_preds(network, inputs, labels),
                            global_step=epoch)
        # Validation ##################################===============###########=========########
        if (epoch+1) % cfg_run["val_interval"] == 0:
            LossMeter = AverageMeter()
            Top1Meter = AverageMeter()
            end = time.time()
            network.eval()
            with torch.no_grad():
                for inputs, labels in dataloader.valid_loader:
                    if print_label_val == 0 :
                        print_label_val = 1
                        print(labels[:20])
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    ## COMPUTE
                    output = network(inputs)
                    labels = labels.to(torch.long)
                    loss = criterion(output, labels)

                    output = output.float()
                    loss = loss.float()
                    ## METRICS
                    prec1 = accuracy(output.data, labels)[0]
                    LossMeter.update(loss.item(), inputs.size(0))
                    Top1Meter.update(prec1.item(), inputs.size(0))

                    TimeMeter.update(time.time() - start)
                    end = time.time()
                    i += 1
            print(f'**[Epoch {epoch+1}/{cfg_run["epoch"]}] [VAL] Loss {LossMeter.avg:.4f})\t Acc: {Top1Meter.avg:.3f})\t TotalTime: {round(TimeMeter.val/3600)}h {round(TimeMeter.val%3600/60)}min')
            print()
            ##### plot
            writer.add_scalar('Validation loss',
                LossMeter.avg,
                epoch )
            writer.add_figure('validation prediction',
                            plot_classes_preds(network, inputs, labels),
                            global_step=epoch )
    return best_network


def test(dataloader, network, criterion, device):
    """
    test accuracy 반환, 여기서도 dataset, dataloader는 dictionary type이다.
    """
    LossMeter = AverageMeter()
    Top1Meter = AverageMeter()
    network.eval()

    # Do validation with test dataset
    running_loss = 0.0
    running_corrects = 0

    i = 0
    print_label = 0
    with torch.no_grad():
        for inputs, labels in dataloader.test_loader:
            if print_label == 0 :
                print_label = 1
                print(labels[:20])
            inputs = inputs.to(device)
            labels = labels.to(device)
            ## COMPUTE
            output = network(inputs)
            labels = labels.to(torch.long)
            loss = criterion(output, labels)

            output = output.float()
            loss = loss.float()
            ## METRICS
            prec1 = accuracy(output.data, labels)[0]
            LossMeter.update(loss.item(), inputs.size(0))
            Top1Meter.update(prec1.item(), inputs.size(0))
            i+=1

    print(f'**[TEST] Loss {LossMeter.avg:.4f})\t Acc: {Top1Meter.avg:.3f})\t')
    print()

    return Top1Meter.avg


def is_cuda():
    if torch.cuda.is_available():
        return "cuda"
    else:
        print("[ DEVICE  ] No CUDA. Working on CPU.")
        return "cpu"


def get_loss(cfg_run):
    return {
        "crossentropy": nn.CrossEntropyLoss()
    }[cfg_run["loss"]]


def get_optimizer(cfg_run, network):
    name = cfg_run["optimizer"]["name"]
    if name == "adam":
        return optim.Adam(network.parameters(), lr=cfg_run["optimizer"]["lr"])
    if name == "sgd":
        return optim.SGD(network.parameters(), cfg_run["optimizer"]["lr"],
                                momentum=cfg_run["optimizer"]["momentum"],
                                weight_decay=cfg_run["optimizer"]["weight_decay"])

    print("WARNING: The name of optimizer is not correct")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
