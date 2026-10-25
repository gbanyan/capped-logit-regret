Three raw-head models,development seed222000 only (not a confirmation seed):
CIFAR100 CIFAR-adapted ResNet18 (3x3 stride1 stem,no initial maxpool,standard
BasicBlocks2/2/2/2,width64/128/256/512); FashionMNIST CNN; FashionMNIST MLP.
ResNet architecture reference: original residual-network method and official
https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/models/resnet.py
Own implementation,not a new architecture. AMP API checked in official PyTorch
2.8 docs https://docs.pytorch.org/docs/2.8/amp.html to match existing runtime.

Train/validation splits come ONLY from official training data, with LAST50 per
fine class for CIFAR(45k/5k) and LAST600 per class for Fashion(54k/6k). All official
test images,including previously unused CIFAR half,remain unevaluated here.
Normalization means/std use training subset only. Save exact split hashes.

CIFAR:100 epochs,SGD lr.1,momentum.9,weight decay.0005,batch128,cosine to.0001;
GPU random reflect-pad4/crop32/horizontal-flip augmentation. Fashion:30 epochs,
Adam lr.001,weight decay.0001,batch256,cosine to.00001,no image augmentation.
Fashion CNN32/64/128 convs+256 hidden/dropout.25; MLP512/256/dropout.2.
AMP FP16 training with GradScaler; validation/replay FP32. TF32 disabled,
deterministic algorithms,paired RNG state retained. No validation-driven early
stopping or best-checkpoint selection; final fixed epoch is the outcome.

Adequacy diagnostics: final validation accuracy>=.70(CIFAR ResNet),>=.92(Fashion
CNN),>=.89(Fashion MLP). Failure stays failure; any repair is a new prospective
round,not a lowered threshold or test-driven schedule change. Development models
never count as five independently frozen confirmation seeds. They supply runtime,
recipe/learnability evidence and potentially fixed representations,explicitly
labeled development. New confirmation matrix is frozen after review/diagnosis.

