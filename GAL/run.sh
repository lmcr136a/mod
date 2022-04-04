resnet56(){
PRETRAINED_RESNET=/home/workspace/nh/mod/models/pretrained/resnet56_cifar10.pt
MIU=1
LAMBDA=0.6
python main.py \
--teacher_dir $PRETRAINED_RESNET --title r56\
--arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU'_test' --gpus 6 \
--data_dir /hdd1/hdd_A/data/cifar10
}

resnet110(){
PRETRAINED_RESNET=/home/workspace/nh/mod/models/pretrained/resnet110_cifar10.pt
MIU=1
LAMBDA=0.6
python main.py \
--teacher_dir $PRETRAINED_RESNET --title r110\
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

googlenet(){
PRETRAINED_GOOGLENET=[pre-trained model dir]
MIU=1e-1
LAMBDA=1e-2
PRINT=200
python main.py \
--teacher_dir $PRETRAINED_GOOGLENET \
--arch googlenet --teacher_model googlenet --student_model googlenet_sparse \
--lambda $LAMBDA --miu $MIU  \
--job_dir 'experiment/googlenet/lambda_'$LAMBDA'_miu_'$MIU \
--train_batch_size 64 --gpus 1
} 

densenet(){
PRETRAINED_DENSENET=[pre-trained model dir]
MIU=1e-1
LAMBDA=1e-2
python main.py \
--teacher_dir $PRETRAINED_DENSENET \
--arch densenet --teacher_model densenet_40 --student_model densenet_40_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/densenet/lambda_'$LAMBDA'_miu_'$MIU \
--train_batch_size 64 --gpus 2 
} 

finetune(){
ARCH=resnet
TITLE=finetune
python finetune.py \
--arch $ARCH --lr 1e-5 \
--title $TITLE \
--refine experiments/r56/resnet_pruned_8.pt \
--job_dir experiment/$ARCH/$TITLE/ \
--pruned 
}


# Training
# vgg;
# resnet56;
# resnet110;
# googlenet;
# densenet;
# Fine-tuning
finetune;