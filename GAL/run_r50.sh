resnet50(){
PRETRAINED_RESNET=/home/workspace/nh/mod/models/pretrained/resnet50_cifar10.pt
MIU=1
LAMBDA=0.49
python main.py \
--teacher_dir $PRETRAINED_RESNET \
--arch resnet --teacher_model resnet_50 --student_model resnet_50_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU'_test' --gpus 7 \
--data_dir /hdd1/hdd_A/data/cifar10
}

resnet110(){
PRETRAINED_RESNET=/home/workspace/nh/mod/models/pretrained/resnet110_cifar10.pt
MIU=1
LAMBDA=0.49
python main.py \
--teacher_dir $PRETRAINED_RESNET \
--arch resnet --teacher_model resnet_110 --student_model resnet_110_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU'_test' --gpus 6 \
--data_dir /hdd1/hdd_A/data/cifar10
}

vgg(){
PRETRAINED_VGG=[pre-trained model dir]
MIU=1e-1
LAMBDA=1e-3
python main.py \
--teacher_dir $PRETRAINED_VGG \
--arch vgg --teacher_model vgg_16_bn --student_model vgg_16_bn_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/vgg/lambda_'$LAMBDA'_miu_'$MIU 
} 


finetune(){
ARCH=resnet
TITLE=r56c10test
python finetune.py \
--arch $ARCH --lr 1e-5 \
--title $TITLE \
--refine experiments/r56/resnet_pruned_8.pt \
--job_dir experiment/$ARCH/$TITLE/ \
--pruned 
}


# Training
# vgg;
resnet50;
# resnet110;
# googlenet;
# densenet;
# Fine-tuning
# finetune;