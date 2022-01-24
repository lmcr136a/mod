import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from utils.metrics import AverageMeter, accuracy
from figure import plot_classes_preds
from torchsummaryX import summary
from lasso import AssembleNetResNet
from utils.utils import show_test_acc


def run(dataloader, network, cfg_run, writer, n_class):
    """
    한방에 train/val을 진행, validation accuracy가 가장 높은 파라미터를 가지고
    test 진행. test accuracy 를 포함한 학습과정 전체를 가지고 있는 정보를 반환.

    Args: dataset, dataloader, network, configuration with run

    Output: history of the run & classification test accuracy
    """

    device = is_cuda()
    network = network.to(device)

    criterion = get_loss(cfg_run)
    optimizer = get_optimizer(cfg_run, network)

    print(f"[LOSS FUNC] {criterion}  [OPTIMIZER] {optimizer}")

    if cfg_run["load_state"]:
        print("\n====================== LOADING STATES... =====================")
        network.load_state_dict(torch.load(cfg_run["load_state"]))
    else:
        print("\n====================== TRAINING START! =====================")
        network = trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer)
        torch.save(network.state_dict(), writer.log_dir+"/best_model.pt")


    if cfg_run.get("lasso", None):
        show_test_acc(test(dataloader, network, criterion, device))

        agent = AssembleNetResNet(cfg_run, dataloader, network, optimizer, criterion, n_class)
        agent.init_graph(pretrained=False)
        summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
        if cfg_run["lasso"]["all_classes"]:
            agent.compress(writer.log_dir, method="lasso", k=cfg_run["lasso"].get("k", 0.49))
        else:
            agent.lasso_compress(cfg_run["lasso"], writer.log_dir)
            
        summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
        show_test_acc(test(dataloader, network, criterion, device))


    network = trainNval(dataloader, network, cfg_run, criterion, optimizer, device, writer)
    test_accuracy = test(dataloader, network, criterion, device)
    
    return test_accuracy


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
            print(f'[Epoch {epoch+1}/{cfg_run["epoch"]}] [TRAIN] Loss {LossMeter.val:.4f} ({LossMeter.avg:.4f})\t Acc: {Top1Meter.val:.3f} ({Top1Meter.avg:.3f})\t Time {BetchTimeMeter.val:.3f} ({BetchTimeMeter.avg:.3f})')

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
            print(f'**[Epoch {epoch+1}/{cfg_run["epoch"]}] [VAL] Loss {LossMeter.val:.4f} ({LossMeter.avg:.4f})\t Acc: {Top1Meter.val:.3f} ({Top1Meter.avg:.3f})\t TotalTime: {round(TimeMeter.val/3600)}h {round(TimeMeter.val%3600/60)}min')
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

    print(f'**[TEST] Loss {LossMeter.val:.4f} ({LossMeter.avg:.4f})\t Acc: {Top1Meter.val:.3f} ({Top1Meter.avg:.3f})\t')
    print()

    return Top1Meter.avg


def is_cuda():
    if torch.cuda.is_available():
        print("[ DEVICE  ] CUDA available")
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
        return optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    print("WARNING: The name of optimizer is not correct")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
