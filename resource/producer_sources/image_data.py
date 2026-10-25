"""Official TRAIN-only image reader;shared by controlled full-pixel producers."""
import gzip
import hashlib
import json
import tarfile
import numpy as np


def training_images(architecture,data_root,cifar):
    if architecture=='resnet18':
        assert hashlib.md5(cifar.read_bytes()).hexdigest()=='03b5dce01913d631647c71ecec9e9cb8'
        with tarfile.open(cifar) as tar:
            raw=np.frombuffer(tar.extractfile('cifar-100-binary/train.bin').read(),dtype=np.uint8).reshape(-1,3074)
        return raw[:,2:].copy().reshape(-1,3,32,32),raw[:,1].astype(np.int64)
    receipt=json.loads((data_root/'input_receipt.json').read_text())
    for record in receipt['fashion']:
        if record['name'].startswith('train'):
            assert hashlib.md5((data_root/'fashion'/record['name']).read_bytes()).hexdigest()==record['md5']
    images=np.frombuffer(gzip.decompress((data_root/'fashion/train-images-idx3-ubyte.gz').read_bytes()),dtype=np.uint8,offset=16).reshape(-1,1,28,28).copy()
    labels=np.frombuffer(gzip.decompress((data_root/'fashion/train-labels-idx1-ubyte.gz').read_bytes()),dtype=np.uint8,offset=8).astype(np.int64)
    return images,labels
