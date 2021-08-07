import torch
import torch.nn as nn
from torch.nn.modules import transformer
from torchvision import models , transforms

class Img2VecEncoder(nn.Module):

    def __init__(self, feature_size=512) -> None:
        super(Img2VecEncoder, self).__init__()
        self.numberFeatures = feature_size
        self.model = models.resnet18(pretrained=True)
        self.featureLayer = self.model._modules.get('avgpool')
        self.normalize =  transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    
    def forward(self, img, ):
        embedding = torch.zeros(
            1, self.numberFeatures, 1 , 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(img)
        h.remove()
        return embedding[0, : , 0 , 0]



def loss_similarity_func(img_vec1, img_vec2, loss_fn=nn.CosineSimilarity()):
    return loss_fn(img_vec1, img_vec2)