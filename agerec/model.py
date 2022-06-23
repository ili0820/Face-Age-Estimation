import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class HealthModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.embeddings = nn.Embedding(2,self.args.hidden_dim)
        self.bn1=nn.BatchNorm1d(self.args.hidden_dim*2)
        self.bn2=nn.BatchNorm1d(21)
        self.bn3=nn.BatchNorm1d(21+self.args.hidden_dim*self.args.batch_size)

        self.layer1=nn.Sequential(
            nn.Linear(self.args.hidden_dim*2+21, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

            
        )
        self.layer2=nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            
        )
        self.layer3=nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            
        )
        self.layer4=nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            
        )
        if self.args.goal=="age":
           self.FFlayer=nn.Linear(128,9)
        else:
            self.FFlayer=nn.Linear(128,1)
        self.softmax=nn.Softmax(dim=1)
        

    def forward(self,cont,cate):


        x1 = [self.embeddings(x) for x in cate]
        x1 = torch.cat(x1).reshape(cate.size()[0],-1)

        x1 = self.bn1(x1)
        x2 = cont
        x2 = self.bn2(x2)

        x = torch.cat([x1, x2], 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.args.goal=="age":
            x=self.softmax(self.FFlayer(x))
        else:
            x=torch.sigmoid(self.FFlayer(x))
        return x


