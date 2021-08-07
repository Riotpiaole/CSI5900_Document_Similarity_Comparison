import time

import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
import torch.optim as optim
from dataset.dataset_constants import ROOT_DIR, TRAIN , TEST , VAL

from pdb import set_trace
from collections import defaultdict
import pandas as pd

vgg16 = models.vgg16()

def flush_data_progress(model, epochs, df, tag='train'):
    torch.save(model.state_dict(), f"./outs/model_state_dict/{epochs}_model_.pt")
    df = pd.DataFrame(df)
    df.to_csv(f"./outs/{tag}_loss.csv",index=False)

def train(model, num_epochs, dataloaders, criterion, device = torch.device('cpu')):
    since = time.time()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad , model.parameters()), 
        lr=0.001, momentum=0.9)
    
    loss_train , loss_val , corrected_pred , acc_val = ( 0 for i in range(4))
    
    avg_loss = 0
    avg_acc = 0
    states = defaultdict(lambda : [])
    
    model = model.to(device)
    criterion = criterion.to(device)
    model.train(True)
    
    train_size , val_size  = len(dataloaders[TRAIN]) , len(dataloaders[VAL]) 

    for epoch in range(num_epochs):
        for batch , (src, target, y) in enumerate(dataloaders[TRAIN]):
            if (time.time() - since )//1000 % 18000000 == 0:
                flush_data_progress(model, epoch,states)
            inputs = src.to(device)
            target = target.to(device)
            label  = y.to(device)
            
            optimizer.zero_grad()
            
            output = model(inputs, target)
            
            loss = criterion(output, label.t()[0])
            
            loss.backward()

            optimizer.step()

            loss_train += loss.item()

            correct_choice = torch.sum((output > 0.70).float()  == label.t()[0])
            corrected_pred += correct_choice
            
            states['train_loss'].append(loss.item())
            states['train_acc'].append(correct_choice) 
            print(f"trianing {batch}/{len(dataloaders[TRAIN])} with loss {loss.item():.3f} {correct_choice:.3f}/{dataloaders[TRAIN].batch_size} ", end='\r')
        
        flush_data_progress(model,epoch, states)
        avg_loss = loss_train /len(dataloaders[TRAIN])
        avg_acc = corrected_pred /len(dataloaders[TRAIN])

        print(f"Complete the training and the loss is {loss_train/len(dataloaders[TRAIN]):.3f}"
            f"\n\t with accuracy {corrected_pred/len(dataloaders[TRAIN]):.3f}")
        model.train(False)
        model.eval()
        
        for batch, (src, target , y)  in enumerate(dataloaders[VAL]):
            inputs = src.to(device)
            target = target.to(device)
            
            label = y.to(device)
            
            with torch.no_grad():
                output = model(inputs, target)
                loss = criterion(output, label)
            
            loss_val += loss.data[0]
            correct_choices = torch.sum((output > 0.70).float()  == label.t()[0])/len(dataloaders[TRAIN])
            acc_val += correct_choices
            
            states['val_loss'].append(loss.item())
            states['val_acc'].append(correct_choices)
            
            print(f"validation {batch}/{len(dataloaders[VAL])} with loss {loss.item():.3f} {correct_choices :.3f}/{dataloaders[VAL].batch_size} ", end='\r')
        
        flush_data_progress(model,epoch, states, tag='val')
        
        avg_loss_val = loss_val / len(dataloaders[VAL])
        avg_acc_val = acc_val / len(dataloaders[VAL])
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
    
    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    return model

if __name__ == "__main__":
    from metrics.bi_direction_models import ConcatedBiModelFeatureExtractor
    from dataset.income_tax_1988 import (
        create_combination_dataloader, 
        generate_dataset,
    )
    df = generate_dataset()
    dataloaders = create_combination_dataloader( df, batch_size=256 )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    from torchvision.models import squeezenet1_1
    model = ConcatedBiModelFeatureExtractor(squeezenet1_1)
    biLoss = torch.nn.BCELoss()
    train(model, 1, dataloaders, biLoss , device=device)