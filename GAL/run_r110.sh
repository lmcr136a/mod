
resnet110(){
PRETRAINED_RESNET=/home/workspace/nh/mod/models/pretrained/resnet110_cifar10.pt
MIU=1
LAMBDA=0.49
python main.py \
--teacher_dir $PRETRAINED_RESNET \
--arch resnet --teacher_model resnet_110 --student_model resnet_110_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU'_test' --gpus 7 \
--data_dir /hdd1/hdd_A/data/cifar10
}

finetune(){
ARCH=resnet
TITLE=finetune
python finetune.py \
--arch $ARCH --lr 1e-5 \
--title $TITLE \
--refine experiment/$ARCH/lambda_0.6_miu_1_test/resnet_pruned_27.pt \
--job_dir experiment/$ARCH/$TITLE/ \
--pruned 
}


# Training
# vgg;
# resnet56;
resnet110;
# googlenet;
# densenet;
# Fine-tuning
# finetune;