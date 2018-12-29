# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import print_function
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet import gluon
from mxnet.base import MXNetError
from mxnet import image
from mxnet.gluon.data.vision import transforms
from mxnet.test_utils import assert_almost_equal
from mxnet.test_utils import almost_equal
from common import assertRaises, setup_module, with_seed, teardown


@with_seed()
def test_to_tensor():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(out_nd.asnumpy(), np.transpose(
        data_in.astype(dtype=np.float32) / 255.0, (2, 0, 1)))


@with_seed()
def test_normalize():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    data_in = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    out_nd = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))(data_in)
    data_expected = data_in.asnumpy()
    data_expected[:][:][0] = data_expected[:][:][0] / 3.0
    data_expected[:][:][1] = (data_expected[:][:][1] - 1.0) / 2.0
    data_expected[:][:][2] = data_expected[:][:][2] - 2.0
    assert_almost_equal(data_expected, out_nd.asnumpy())


# @with_seed()
# def test_resize():
#     def _test_resize_with_diff_type(dtype):
#         # test normal case
#         data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
#         out_nd = transforms.Resize(200)(data_in)
#         data_expected = image.imresize(data_in, 200, 200, 1)
#         assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
#         # test 4D input
#         data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
#         out_batch_nd = transforms.Resize(200)(data_bath_in)
#         for i in range(len(out_batch_nd)):
#             assert_almost_equal(image.imresize(data_bath_in[i], 200, 200, 1).asnumpy(),
#                 out_batch_nd[i].asnumpy())
#         # test interp = 2
#         out_nd = transforms.Resize(200, interpolation=2)(data_in)
#         data_expected = image.imresize(data_in, 200, 200, 2)
#         assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
#         # test height not equals to width
#         out_nd = transforms.Resize((200, 100))(data_in)
#         data_expected = image.imresize(data_in, 200, 100, 1)
#         assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
#         # test keep_ratio
#         out_nd = transforms.Resize(150, keep_ratio=True)(data_in)
#         data_expected = image.imresize(data_in, 150, 225, 1)
#         assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
#         def _test_size_below_zero_Exception():
#             transforms.Resize(-150, keep_ratio=True)(data_in)
#         assertRaises(MXNetError, _test_size_below_zero_Exception)
#         def _test_size_more_than_2_Exception():
#             transforms.Resize((100, 100, 100), keep_ratio=True)(data_in)
#         assertRaises(MXNetError, _test_size_more_than_2_Exception)

#     for dtype in ['uint8', 'int8', 'float32', 'float64']:
#         _test_resize_with_diff_type(dtype)    


@with_seed()
def test_crop():
    def _test_crop_with_diff_type(dtype):
        # test normal case
        data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
        out_nd = transforms.Crop(0, 0, 100, 50)(data_in)
        data_expected = image.fixed_crop(data_in, 0, 0, 100, 50)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test 4D input
        data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
        out_batch_nd = transforms.Crop(0, 0, 100, 50)(data_bath_in)
        for i in range(len(out_batch_nd)):
            assert_almost_equal(image.fixed_crop(data_bath_in[i], 0, 0, 100, 50).asnumpy(),
                out_batch_nd[i].asnumpy())
        # test normal case with resize
        data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
        out_nd = transforms.Crop(0, 0, 100, 50, (25, 25), 2)(data_in)
        data_expected = image.imresize(image.fixed_crop(data_in, 0, 0, 100, 50), 25, 25, 2)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test 4D input with resize
        data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
        out_batch_nd = transforms.Crop(0, 0, 100, 50, (25, 25), 2)(data_bath_in)
        for i in range(len(out_batch_nd)):
            assert_almost_equal(image.imresize(image.fixed_crop(data_bath_in[i], 0, 0, 100, 50), 25, 25, 2).asnumpy(),
                out_batch_nd[i].asnumpy())
        def _test_size_below_zero_Exception():
            transforms.Crop(0, 0, 100, 50, (-25, 25), 2)(data_in)
        assertRaises(MXNetError, _test_size_below_zero_Exception)    
        def _test_height_and_width_below_zero_Exception():
            transforms.Crop(0, 0, -100, -50)(data_in)
        assertRaises(MXNetError, _test_height_and_width_below_zero_Exception)

    for dtype in ['uint8', 'int8', 'float32', 'float64']:
        _test_crop_with_diff_type(dtype)    

# @with_seed()
# def test_center_crop():
#     def _test_center_crop_with_diff_type(dtype):
#         # test normal case
#         data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
#         out_nd = transforms.CenterCrop((100, 100), 2)(data_in)
#         data_expected = image.center_crop(data_in, (100, 100), 2)[0]
#         assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
#         # test 4D input
#         data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
#         out_batch_nd = transforms.CenterCrop((100, 100), 2)(data_bath_in)
#         for i in range(len(out_batch_nd)):
#             assert_almost_equal(image.center_crop(data_bath_in[i], (100, 100), 2)[0].asnumpy(),
#                 out_batch_nd[i].asnumpy())
#         # test size is larger than input image
#         data_in = nd.random.uniform(0, 255, (100, 100, 3)).astype(dtype)
#         out_nd = transforms.CenterCrop((200, 150), 2)(data_in)
#         data_expected = image.center_crop(data_in, (200, 150), 2)[0]
#         assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
#         # test size is larger than input image 4D
#         data_bath_in = nd.random.uniform(0, 255, (3, 100, 100, 3)).astype(dtype)
#         out_batch_nd = transforms.CenterCrop((200, 150), 2)(data_bath_in)
#         for i in range(len(out_batch_nd)):
#             assert_almost_equal(image.center_crop(data_bath_in[i], (200, 150), 2)[0].asnumpy(),
#                 out_batch_nd[i].asnumpy())
#         def _test_size_below_zero_Exception():
#             transforms.CenterCrop((-200, 150), 2)(data_in)
#         assertRaises(MXNetError, _test_size_below_zero_Exception)

#     for dtype in ['uint8', 'int8', 'float32', 'float64']:
#         _test_center_crop_with_diff_type(dtype)


# @with_seed()
# def test_random_resize_crop():
#     def _test_random_resize_crop_with_diff_type(dtype):
#         # test normal case
#         data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
#         out_nd = transforms.RandomResizedCrop((100, 120))(data_in)
#         assert out_nd.shape == (120, 100, 3), out_nd.shape
#         # test 4D input
#         data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
#         out_batch_nd = transforms.RandomResizedCrop((50, 70))(data_bath_in)
#         assert out_batch_nd.shape == (3, 70, 50, 3), out_nd.shape
#         # test normal case
#         data_in = nd.random.uniform(0, 255, (50, 30, 3)).astype(dtype)
#         out_nd = transforms.RandomResizedCrop((140, 100))(data_in)
#         assert out_nd.shape == (100, 140, 3), out_nd.shape
#          # test size is larger than input image 4D
#         data_in = nd.random.uniform(0, 255, (4, 50, 30, 3)).astype(dtype)
#         out_nd = transforms.RandomResizedCrop((100, 140))(data_in)
#         assert out_nd.shape == (4, 140, 100, 3), out_nd.shape

#     for dtype in ['uint8', 'int8', 'float32', 'float64']:
#         _test_random_resize_crop_with_diff_type(dtype)


@with_seed()
def test_flip_left_right():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    flip_in = data_in[:, ::-1, :]
    data_trans = nd.image.flip_left_right(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(flip_in, data_trans.asnumpy())


@with_seed()
def test_flip_top_bottom():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    flip_in = data_in[::-1, :, :]
    data_trans = nd.image.flip_top_bottom(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(flip_in, data_trans.asnumpy())


@with_seed()
def test_transformer():
    from mxnet.gluon.data.vision import transforms

    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.Resize(300, keep_ratio=True),
        transforms.CenterCrop(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomBrightness(0.1),
        transforms.RandomContrast(0.1),
        transforms.RandomSaturation(0.1),
        transforms.RandomHue(0.1),
        transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])])

    transform(mx.nd.ones((245, 480, 3), dtype='uint8')).wait_to_read()


if __name__ == '__main__':
    import nose
    nose.runmodule()
