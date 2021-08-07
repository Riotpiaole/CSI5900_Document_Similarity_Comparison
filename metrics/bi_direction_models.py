import torch 
import torch.nn as nn
from pdb import set_trace
import torchvision.transforms as transforms

class ConcatedBiModelFeatureExtractor(nn.Module):
    def __init__(self, custom_model, pretrained=True):
        super().__init__()
        self.module1 = custom_model(pretrained=pretrained)
        self.module2 = custom_model(pretrained=pretrained)
        
        for param in self.module1.features.parameters():
            param.requires_grad = False
        
        for param in self.module2.features.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
                
        # self.loss = cca_loss(1, True , next(self.parameters()).device)
        self.loss = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_image, target_image):
        x1 = self.module1(src_image)
        x2 = self.module2(target_image)
        
        similarity = self.loss(x1, x2)
        output = self.sigmoid(similarity)
        return output

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from torchvision.models import vgg16_bn
    from dataset.income_tax_1988 import (
        create_combination_dataloader, 
        generate_dataset,
    )
    from train import train
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    df = generate_dataset()
    dataloaders = create_combination_dataloader(df)
    
    model = ConcatedBiModelFeatureExtractor(vgg16_bn).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # transform_pipeline = \
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])

   