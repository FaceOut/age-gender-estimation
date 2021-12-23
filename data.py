import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

class AgeImageIter(mx.image.ImageIter):
  def next(self):
    origin_batch = super().next()
    data = origin_batch.data
    labels = origin_batch.label[0]

    new_labels = nd.empty((labels.shape[0], 101))
    for i in range(labels.shape[0]):
      label = labels[i].asnumpy()
      gender, age = label.astype(np.int)
      age = min(100, max(1, age))
      plabel = np.zeros(shape=(101,), dtype=np.float32)
      plabel[0] = gender

      plabel[1:age+1] = 1
      
      new_labels[i][:] = plabel
    return mx.io.io.DataBatch(data, [new_labels])
    