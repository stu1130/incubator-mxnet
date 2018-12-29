import mxnet as mx
import numpy as np
from mxnet import image

# with open("flower.jpg", 'rb') as fp:
#     str_image = fp.read()
# image = mx.img.imdecode(str_image, to_rgb=0)
# print(image)

# d = mx.nd.random.uniform(0, 255, (5, 5, 3)).astype(dtype=np.uint8)
# print(mx.nd.moveaxis(d, 2, 0))
# image_nd = image.center_crop(d, (10, 8), 1)[0]
# print(mx.nd.moveaxis(image_nd, 2, 0))
# image = mx.nd.image.center_crop(d, (10, 8), 1)
# transformer = mx.gluon.data.vision.transforms.Crop(0, 0, 1000, 500)
# transformer = mx.gluon.data.vision.transforms.Crop(0, 0, 100, 100, (-50, 50), 1)
# print(mx.nd.moveaxis(image, 2, 0))

# data_in = mx.nd.random.uniform(0, 255, (300, 200, 3)).astype('int32')
# print(mx.nd.moveaxis(data_in, 2, 0))
# out_nd = mx.gluon.data.vision.transforms.Resize(5)(data_in)
# print(mx.nd.moveaxis(out_nd, 2, 0))

d = mx.nd.random.uniform(0, 255, (2, 5, 5, 3)).astype(dtype=np.uint8)
print(mx.nd.moveaxis(d, 3, 1))
transformer = mx.gluon.data.vision.transforms.RandomResizedCrop((4, 4))
# transformer = mx.gluon.data.vision.transforms.RandomContrast(2)
print(mx.nd.moveaxis(transformer(d), 3, 1))
print('=======================')
for i in range(len(d)):
    print(mx.nd.moveaxis(image.random_size_crop(d[i], (4,4),
    (0.08, 1.0), (3.0/4.0, 4.0/3.0), 1)[0], 2, 0))