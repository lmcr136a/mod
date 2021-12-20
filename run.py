import torch
import torch.nn as nn
import torch.optim as optim
from metrics import AverageMeter, accuracy



def run(dataset, dataloader, network, cfg_run):
    """
    한방에 train/val을 진행, validation accuracy가 가장 높은 파라미터를 가지고
    test 진행. test accuracy 를 포함한 학습과정 전체를 가지고 있는 정보를 반환.

    Args: dataset, dataloader, network, configuration with run

    Output: history of the run & classification test accuracy
    """
    device = is_cuda()
    network = network.to(device)

    criterion = get_loss(cfg_run)
    optimizer = get_optimizer(cfg_run)

    print(f"[LOSS FUNC] {criterion}  [OPTIMIZER] {optimizer}")
    print("\n====================== TRAINING START! =====================")

    best_network = trainNval(dataset, dataloader, network, cfg_run, criterion, optimizer, device)

    test_accuracy = test(dataset, dataloader, best_network, criterion, device)
    
    return test_accuracy


def trainNval(dataset, dataloader, network, cfg_run, criterion, optimizer, device):
    """
    cfg_run에 담긴대로 training/validation 진행
    가장 높은 validation accuracy를 가진 네트워크를 출력

    여기서 data는 dictionary type이다.
    """
    best_acc = 0.0
    best_network = None

    for epoch in range(cfg_run["epoch"]):
        BetchTimeMeter = AverageMeter()
        LossMeter = AverageMeter()
        Top1Meter = AverageMeter()
        end = time.time()

        network.train()

        for inputs, labels in dataloader['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            ## COMPUTE
            output = model(input_var)
            loss = criterion(output, target_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            ## METRICS
            prec1 = accuracy(output.data, target)[0]
            LossMeter.update(loss.item(), input.size(0))
            Top1Meter.update(prec1.item(), input.size(0))

            BetchTimeMeter.update(time.time() - end)
            end = time.time()

            if (epoch+1) % cfg_run["print_interval"] == 0:
                print(f'[Epoch {epoch+1}/{cfg_run["epoch"]}] [TRAIN] Loss {LossMeter.val:.4f} ({LossMeter.avg:.4f})\t Acc: {top1.val:.3f} ({top1.avg:.3f})\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # Validation step for every cfg_run["val_interval"] times
        if (epoch+1) % cfg_run["val_interval"] == 0:
            BetchTimeMeter = AverageMeter()
            LossMeter = AverageMeter()
            Top1Meter = AverageMeter()
            end = time.time()

            network.eval()
            with torch.no_grad():
                for inputs, labels in dataloader['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    ## COMPUTE
                    output = model(input_var)
                    loss = criterion(output, target_var)

                    output = output.float()
                    loss = loss.float()
                    ## METRICS
                    prec1 = accuracy(output.data, target)[0]
                    LossMeter.update(loss.item(), input.size(0))
                    Top1Meter.update(prec1.item(), input.size(0))

                    BetchTimeMeter.update(time.time() - end)
                    end = time.time()
            print(f'**[Epoch {epoch+1}/{cfg_run["epoch"]}] [VAL] Loss {LossMeter.val:.4f} ({LossMeter.avg:.4f})\t Acc: {top1.val:.3f} ({top1.avg:.3f})\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})')
            print()
            if top1.avg >= best_acc:
                best_network = network
                best_acc = top1.avg

    return best_network


def test(dataset, dataloader, network, criterion, device):
    """
    test accuracy 반환, 여기서도 dataset, dataloader는 dictionary type이다.
    """
    network.eval()

    # Do validation with test dataset
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            ## COMPUTE
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()
            ## METRICS
            prec1 = accuracy(output.data, target)[0]
            LossMeter.update(loss.item(), input.size(0))
            Top1Meter.update(prec1.item(), input.size(0))

            BetchTimeMeter.update(time.time() - end)
            end = time.time()
    print(f'**[TEST] Loss {LossMeter.val:.4f} ({LossMeter.avg:.4f})\t Acc: {top1.val:.3f} ({top1.avg:.3f})\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})')
    print()

    return test_acc.item()


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
    weight_decay = cfg_run["optimizer"]["weight_decay"]
    if name == "adam":
        return optim.Adam(network.parameters(), lr=cfg_run["optimizer"]["lr"])
    if name == "SGD":
        return optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    print("WARNING: The name of optimizer is not correct")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
