import os

if __name__ == "__main__":
    targets = ["./data/inversion/cifar100_r20_196/train/", "./data/inversion/cifar100_r34_196/final_images/"]

    for target in targets:
        # os.chdir(target)
        for dirname in [name for name in os.listdir(target)]:
            os.rename(target+dirname, target+dirname[1:])
