"""Validation-only development baselines; official test images never evaluated."""
import argparse
import gzip
import hashlib
import json
import math
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Block(nn.Module):
    def __init__(self, before, after, stride=1):
        super().__init__()
        self.path = nn.Sequential(nn.Conv2d(before, after, 3, stride, 1, bias=False),
            nn.BatchNorm2d(after), nn.ReLU(), nn.Conv2d(after, after, 3, 1, 1, bias=False), nn.BatchNorm2d(after))
        self.skip = nn.Identity() if before == after and stride == 1 else nn.Sequential(
            nn.Conv2d(before, after, 1, stride, bias=False), nn.BatchNorm2d(after))

    def forward(self, x):
        return F.relu(self.path(x) + self.skip(x))


class ResNet18(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()]
        before = 64
        for stage, after in enumerate((64, 128, 256, 512)):
            for block in range(2):
                layers.append(Block(before, after, 2 if stage and block == 0 else 1))
                before = after
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.output = nn.Linear(512, classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.output(self.features(x))


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.output = nn.Sequential(nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(.25), nn.Linear(256, 10))

    def forward(self, x):
        return self.output(self.features(x))


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
            nn.Dropout(.2), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.2))
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(self.features(x))


class GeometryModel(nn.Module):
    def __init__(self, base, head, classes, beta):
        super().__init__()
        self.base, self.head = base, head
        ratio = (classes - 2) * (1 - beta) / (2 * beta)
        assert ratio > 1
        self.cap = math.log(ratio) / 2
        self.radius = math.sqrt(2) * self.cap

    def forward(self, x):
        raw = self.base(x)
        if self.head == 'raw':
            return raw
        if self.head == 'box':
            return raw.clamp(-self.cap, self.cap)
        if self.head == 'tanh':
            return self.cap * torch.tanh(raw / self.cap)
        if self.head == 'ball':
            return raw * (self.radius / raw.norm(dim=1, keepdim=True).clamp_min(1e-20)).clamp(max=1)
        raise ValueError(self.head)


def model_for(name):
    base = {'resnet18': ResNet18, 'fashion_cnn': FashionCNN, 'fashion_mlp': FashionMLP}[name]()
    return GeometryModel(base, ARGS.head, 100 if name == 'resnet18' else 10, ARGS.beta)


def load_training(architecture, data_root, cifar):
    if architecture == 'resnet18':
        assert hashlib.md5(cifar.read_bytes()).hexdigest() == '03b5dce01913d631647c71ecec9e9cb8'
        with tarfile.open(cifar) as tar:
            raw = np.frombuffer(tar.extractfile('cifar-100-binary/train.bin').read(), dtype=np.uint8).reshape(-1, 3074)
        images = raw[:, 2:].copy().reshape(-1, 3, 32, 32); labels = raw[:, 1].astype(np.int64)
        each, classes = 50, 100
    else:
        # Only official TRAIN files; no t10k/test tensor is constructed.
        receipt = json.loads((data_root / 'input_receipt.json').read_text())
        for record in receipt['fashion']:
            if record['name'].startswith('train'):
                assert hashlib.md5((data_root / 'fashion' / record['name']).read_bytes()).hexdigest() == record['md5']
        raw_images = gzip.decompress((data_root / 'fashion/train-images-idx3-ubyte.gz').read_bytes())
        raw_labels = gzip.decompress((data_root / 'fashion/train-labels-idx1-ubyte.gz').read_bytes())
        images = np.frombuffer(raw_images, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).copy()
        labels = np.frombuffer(raw_labels, dtype=np.uint8, offset=8).astype(np.int64)
        each, classes = 600, 10
    valid = np.concatenate([np.flatnonzero(labels == label)[-each:] for label in range(classes)])
    valid.sort(); training = np.setdiff1d(np.arange(len(labels)), valid)
    x = torch.from_numpy(images).cuda().float().div_(255)
    y = torch.from_numpy(labels).cuda()
    train_ids = torch.from_numpy(training).cuda(); val_ids = torch.from_numpy(valid).cuda()
    mean = x[train_ids].mean((0, 2, 3), keepdim=True)
    std = x[train_ids].std((0, 2, 3), keepdim=True).clamp_min(.01)
    x.sub_(mean).div_(std)
    identity = dict(training_examples=len(training), validation_examples=len(valid),
        train_indices_sha256=hashlib.sha256(training.tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
        mean=mean.flatten().cpu().tolist(), std=std.flatten().cpu().tolist(),
        official_test_evaluated=False)
    return x, y, train_ids, val_ids, identity


def augment(x, generator):
    size = len(x)
    pad = F.pad(x, (4, 4, 4, 4), mode='reflect')
    shifts = torch.randint(9, (2, size), device='cuda', generator=generator)
    indices = torch.arange(32, device='cuda')
    rows = (shifts[0, :, None] + indices).view(size, 1, 32, 1).expand(size, 3, 32, 40)
    cropped = pad.gather(2, rows)
    cols = (shifts[1, :, None] + indices).view(size, 1, 1, 32).expand(size, 3, 32, 32)
    cropped = cropped.gather(3, cols)
    flip = torch.rand(size, device='cuda', generator=generator) < .5
    return torch.where(flip[:, None, None, None], cropped.flip(3), cropped)


def evaluate(model, x, y, indices):
    model.eval(); losses, correct, logits = 0., 0, []
    with torch.no_grad():
        for ids in indices.split(256):
            z = model(x[ids])  # FP32 evaluation, no autocast.
            losses += F.cross_entropy(z, y[ids], reduction='sum').item()
            correct += (z.argmax(1) == y[ids]).sum().item()
            logits.append(z.cpu().numpy())
    return dict(CE=losses / len(indices), accuracy=correct / len(indices)), np.concatenate(logits)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    model = model_for(args.architecture).cuda()
    initialization = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    x, y, training, validation, identity = load_training(args.architecture, args.data_root, args.cifar)
    is_cifar = args.architecture == 'resnet18'
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005) if is_cifar else torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0001 if is_cifar else .00001)
    scaler = torch.amp.GradScaler('cuda')
    shuffle = torch.Generator(device='cuda').manual_seed(args.seed + 11)
    aug = torch.Generator(device='cuda').manual_seed(args.seed + 29)
    start = time.monotonic(); history = []
    write(args.output / 'input.json', identity)
    for epoch in range(args.epochs + 1):
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics, logits = evaluate(model, x, y, validation)
            history.append(dict(epoch=epoch, validation=metrics, elapsed_seconds=time.monotonic() - start))
            write(args.output / 'progress.json', dict(architecture=args.architecture, seed=args.seed,
                  epochs=args.epochs, latest=history[-1], official_test_evaluated=False))
            print(json.dumps(history[-1]), flush=True)
        if epoch == args.epochs:
            break
        model.train()
        order = training[torch.randperm(len(training), device='cuda', generator=shuffle)]
        for ids in order.split(128 if is_cifar else 256):
            xb = augment(x[ids], aug) if is_cifar else x[ids]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                objective = F.cross_entropy(model(xb), y[ids])
            scaler.scale(objective).backward(); scaler.step(optimizer); scaler.update()
        scheduler.step()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'model.pt')
    # Retained in task-owned raw storage for a prospectively authorized extension.
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(), shuffle=shuffle.get_state(), augmentation=aug.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), epoch=args.epochs), args.output / 'resume.pt')
    np.savez_compressed(args.output / 'validation.npz', logits=logits, labels=y[validation].cpu().numpy(), indices=validation.cpu().numpy())
    fresh = model_for(args.architecture).cuda()
    fresh.load_state_dict(torch.load(args.output / 'model.pt', weights_only=True, map_location='cpu'))
    replay, replay_logits = evaluate(fresh, x, y, validation)
    assert replay == metrics and np.array_equal(replay_logits, logits)
    write(args.output / 'complete.json', dict(architecture=args.architecture, seed=args.seed, epochs=args.epochs,
        phase='frozen confirmation matrix; official test still sealed', head=args.head, beta=args.beta, history=history, initialization_sha256=initialization,
        parameter_count=sum(p.numel() for p in model.parameters()), torch=torch.__version__, CUDA=torch.version.cuda,
        official_test_evaluated=False, replay_validation_exact=True, seconds=time.monotonic() - start,
        model_sha256=hashlib.sha256((args.output / 'model.pt').read_bytes()).hexdigest(),
        peak_cuda_allocated=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=('resnet18', 'fashion_cnn', 'fashion_mlp'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cifar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=222000)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--head', choices=('raw', 'box', 'tanh', 'ball'), required=True)
    parser.add_argument('--beta', type=float, default=.3)
    ARGS = parser.parse_args()
    main(ARGS)
