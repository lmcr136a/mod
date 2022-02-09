import time
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.utils.prune as prune

from utils.metrics import AverageMeter, accuracy
from figure import plot_classes_preds
from torchsummaryX import summary
from lasso import AssembleNetResNet
from utils.utils import show_test_acc
from dataset import get_dataloader, get_test_dataloader
from models.model import get_network


def run(cfg, writer):
    """
    한방에 train/val을 진행, validation accuracy가 가장 높은 파라미터를 가지고
    test 진행. test accuracy 를 포함한 학습과정 전체를 가지고 있는 정보를 반환.

    Args: dataset, dataloader, network, configuration with run

    Output: history of the run & classification test accuracy
    """
    dataloader, n_class = get_dataloader(cfg, get_only_targets=True)
    test_dataloader, _ = get_test_dataloader(cfg, dataloader, get_only_targets=True)
    network = get_network(cfg["network"]["model"], n_class)

    cfg_run = cfg["run"]


    device = is_cuda()
    np.random.seed(cfg_run["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(cfg_run["seed"])
        device_num = cfg_run.get("gpu_device", 0)
        torch.cuda.set_device(device_num)
        print(f"[ DEVICE  ] CUDA GPU {device_num} available")
    network = network.to(device)

    criterion = get_loss(cfg_run)
    optimizer = get_optimizer(cfg_run, network)

    print(f"[LOSS FUNC] {criterion}  [OPTIMIZER] {optimizer}")

    if cfg["network"].get("load_state", None):
        print("\n====================== LOADING STATES... =====================")
        network.load_state_dict(torch.load(cfg["network"].get("load_state", None)))
    else:
        print("\n====================== TRAINING START! =====================")
        network = trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer)
        torch.save(network.state_dict(), writer.log_dir+"/best_model.pt")

    show_test_acc(test(test_dataloader, network, criterion, device))   ## CIFAR



    if cfg["network"].get("lasso", None):

        agent = AssembleNetResNet(cfg_run, dataloader, network, optimizer, criterion, n_class)   ## INV
        agent.init_graph(pretrained=False)
        summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
        if cfg["network"]["lasso"]["all_classes"]:
            all_class_dataloader, _ = get_dataloader(cfg)
            agent.data_loader = all_class_dataloader
            agent.compress(writer.log_dir, method="lasso", k=cfg["network"]["lasso"].get("k", 0.49))
        else:
            agent.lasso_compress(cfg["network"]["lasso"], writer.log_dir)
            
        summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
        show_test_acc(test(test_dataloader, network, criterion, device))   ## CIFAR

        network = trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer)   ## INV
        show_test_acc(test(test_dataloader, network, criterion, device))   ## CIFAR
    


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
    for epoch in range(cfg_run["epoch"]):
        BetchTimeMeter = AverageMeter()
        LossMeter = AverageMeter()
        Top1Meter = AverageMeter()
        end = time.time()

        network.train()

        for inputs, labels in dataloader.train_loader:
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
            print(f'**[Epoch {epoch+1}/{cfg_run["epoch"]}] [VAL] Loss {LossMeter.avg:.4f})\t Acc: {Top1Meter.avg:.3f})\t TotalTime: {round(TimeMeter.val/3600)}h {round(TimeMeter.val%3600/60)}min')
            print()
            ##### plot
            writer.add_scalar('Validation loss',
                LossMeter.avg,
                epoch )
            writer.add_figure('validation prediction',
                            plot_classes_preds(network, inputs, labels),
                            global_step=epoch )
            if Top1Meter.avg >= best_acc:
                best_network = network
                best_acc = Top1Meter.avg
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

    with torch.no_grad():
        for inputs, labels in dataloader.test_loader:
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
    if name == "SGD":
        return optim.SGD(network.parameters(), cfg_run["optimizer"]["lr"],
                                momentum=cfg_run["optimizer"]["momentum"],
                                weight_decay=cfg_run["optimizer"]["weight_decay"])

    print("WARNING: The name of optimizer is not correct")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
