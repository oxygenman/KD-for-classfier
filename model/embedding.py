import torch.nn as nn
import torch.nn.functional as F
from  model.backbone.shufflenetv2 import ShuffleNet
__all__ = ["LinearEmbedding"]


class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=200, normalize=True):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, x, get_ha=False):
        if get_ha:
            b1, b2, b3, b4, pool,out= self.base(x,True)
        else:
            pool,out = self.base(x)
        if isinstance(self.base, ShuffleNet):
            pool=pool.mean([2,3])
        else:
            pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        if get_ha:
            return b1, b2, b3, b4, pool,embedding

        return pool, embedding
