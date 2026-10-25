"""Final official-test reader;never imported by a training producer."""
import gzip
import hashlib
import json
import struct
import tarfile

import numpy as np


def test_images(architecture,data_root,cifar):
    """Return the predeclared untouched test subset and its original indices."""
    if architecture=='resnet18':
        md5=hashlib.md5(cifar.read_bytes()).hexdigest()
        assert md5=='03b5dce01913d631647c71ecec9e9cb8'
        with tarfile.open(cifar) as tar:
            contents=tar.extractfile('cifar-100-binary/test.bin').read()
        raw=np.frombuffer(contents,dtype=np.uint8).reshape(-1,3074)
        assert len(raw)==10000
        labels=raw[:,1].astype(np.int64)
        assert all((labels==c).sum()==100 for c in range(100))
        indices=np.concatenate([np.flatnonzero(labels==c)[-50:] for c in range(100)])
        indices.sort();assert len(indices)==5000
        images=raw[indices,2:].copy().reshape(-1,3,32,32)
        return images,labels[indices],indices,dict(dataset='CIFAR100',archive_md5=md5,
            member='cifar-100-binary/test.bin',member_sha256=hashlib.sha256(contents).hexdigest(),
            selection='last50 original test indices per fine class',examples=5000)
    assert architecture in ('fashion_cnn','fashion_mlp')
    receipt=json.loads((data_root/'input_receipt.json').read_text());records=[];decoded={}
    for name in ('t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'):
        record=next(r for r in receipt['fashion'] if r['name']==name)
        contents=(data_root/'fashion'/name).read_bytes()
        assert hashlib.md5(contents).hexdigest()==record['md5']
        assert hashlib.sha256(contents).hexdigest()==record['sha256']
        decoded[name]=gzip.decompress(contents)
        records.append(dict(name=name,md5=record['md5'],sha256=record['sha256']))
    ri,rl=(decoded[name] for name in ('t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'))
    assert struct.unpack('>IIII',ri[:16])==(2051,10000,28,28) and len(ri)==16+10000*784
    assert struct.unpack('>II',rl[:8])==(2049,10000) and len(rl)==8+10000
    images=np.frombuffer(ri,dtype=np.uint8,offset=16).reshape(-1,1,28,28).copy()
    labels=np.frombuffer(rl,dtype=np.uint8,offset=8).astype(np.int64)
    return images,labels,np.arange(10000),dict(dataset='FashionMNIST',files=records,
        selection='all official10000 test examples',examples=10000)
