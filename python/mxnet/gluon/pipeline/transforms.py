from ..block import Block, HybridBlock
from ..nn import Sequential, HybridSequential
    
class Compose(HybridSequential):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.add(hybrid[0])
                hybrid = []
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                hblock.hybridize()
                self.add(hblock)
                hybrid = []

            if i is not None:
                self.add(i)
    
class Normalize(HybridBlock):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self._mean = mean
        self._std = std

    def hybrid_forward(self, F, x):
        return F.image.normalize(x, self._mean, self._std)

class ToTensor(HybridBlock):
    def __init__(self):
        super(ToTensor, self).__init__()

    def hybrid_forward(self, F, x):
        return F.image.to_tensor(x)