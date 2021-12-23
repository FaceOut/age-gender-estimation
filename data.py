import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

from mxnet.util import is_np_array
from mxnet import numpy as _mx_np  

class AgeImageIter(mx.image.ImageIter):
    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 last_batch_handle='pad', **kwargs):
        super().__init__(batch_size, data_shape, label_width,
                path_imgrec, path_imglist, path_root, path_imgidx,
                shuffle, part_index, num_parts, aug_list, imglist,
                data_name, label_name, dtype,
                last_batch_handle, **kwargs)
        self.provide_label = [(label_name, (batch_size, 101))]
        self.provide_origin_label = [(label_name, (batch_size, label_width))]

    def next(self):
        origin_batch = self.__origin_patched_next()
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

    # origin mxnet imageiternext
    def __origin_patched_next(self):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        # if last batch data is rolled over
        if self._cache_data is not None:
            # check both the data and label have values
            assert self._cache_label is not None, "_cache_label didn't have values"
            assert self._cache_idx is not None, "_cache_idx didn't have values"
            batch_data = self._cache_data
            batch_label = self._cache_label
            i = self._cache_idx
            # clear the cache data
        else:
            if is_np_array():
                zeros_fn = _mx_np.zeros
                empty_fn = _mx_np.empty
            else:
                zeros_fn = nd.zeros
                empty_fn = nd.empty
            batch_data = zeros_fn((batch_size, c, h, w))
            batch_label = empty_fn(self.provide_origin_label[0][1])
            i = self._batchify(batch_data, batch_label)
        # calculate the padding
        pad = batch_size - i
        # handle padding for the last batch
        if pad != 0:
            if self.last_batch_handle == 'discard':
                raise StopIteration
            # if the option is 'roll_over', throw StopIteration and cache the data
            if self.last_batch_handle == 'roll_over' and \
                self._cache_data is None:
                self._cache_data = batch_data
                self._cache_label = batch_label
                self._cache_idx = i
                raise StopIteration

            _ = self._batchify(batch_data, batch_label, i)
            if self.last_batch_handle == 'pad':
                self._allow_read = False
            else:
                self._cache_data = None
                self._cache_label = None
                self._cache_idx = None

        return mx.io.io.DataBatch([batch_data], [batch_label], pad=pad)