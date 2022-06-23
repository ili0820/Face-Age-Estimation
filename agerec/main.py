from args import parse_args
from dataset import HealthDataset
from model import HealthModel
from utils import get_dataloader,get_optimizer

import pandas as pd
import os
import random
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def setSeeds(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    # wandb.login()

    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device


    # kf = KFold(n_splits=args.n_kfold)
    # for train_index, test_index in kf.split(train_data):
    #     train_dataset, valid_dataset = train_data[train_index], train_data[test_index]
    
    train_data=HealthDataset(args,args.train_dir)
    train,valid=train_test_split(train_data,test_size=0.2)
    test=HealthDataset(args,args.test_dir)


    train_loader,valid_loader,test_loader=get_dataloader(train,valid,test,args)

    #train mode
    if args.mode=="train":
        best_acc = -1
        early_stopping_counter = 0
        #multi class
        if args.goal=="age":
            criterion = torch.nn.CrossEntropyLoss()
            model=HealthModel(args)
            optimizer=get_optimizer(model,args)
        #binary
        else:
            criterion = torch.nn.BCELoss()
            model=HealthModel(args)
            optimizer=get_optimizer(model,args)
        for epoch in range(args.n_epochs):
            ###train
            model.train()
            train_labels=[]
            train_preds=[]
            total_loss=0
            for step, batch in enumerate(train_loader):
                cont,cate,label=batch
                cont.to(args.device)
                cate.to(args.device)

                preds=model(cont,cate)
    
                if args.goal =="age":
                    loss=criterion(preds,label.view(-1))
                else:
                    loss=criterion(preds,label)


                total_loss+=float(loss)

                train_labels.extend(label)
                train_preds.extend(preds.detach())

                loss.backward()
                optimizer.step()
            if args.goal =="age":
                train_preds=[np.argmax(x) for x in train_preds]
            else:
                train_preds=[1 if x >=0.5 else 0 for x in train_preds]
            acc= accuracy_score(torch.cat(train_labels), train_preds)

            ##validation
            model.eval()
            valid_labels=[]
            valid_preds=[]
            for step, batch in enumerate(valid_loader):
                cont,cate,label=batch
                cont.to(args.device)
                cate.to(args.device)

                preds=model(cont,cate)
                valid_labels.extend(label)
                valid_preds.extend(preds.detach())
            if args.goal =="age":
                valid_preds=[np.argmax(x) for x in valid_preds]
            else:
                valid_preds=[1 if x >=0.5 else 0 for x in valid_preds]
            v_acc= accuracy_score(torch.cat(valid_labels),valid_preds)

            if v_acc > best_acc:
                best_acc=v_acc
                torch.save(model.state_dict(), args.model_dir+args.goal+args.model_name)
            else:
                early_stopping_counter+=1
                print("Early Stop Count:",early_stopping_counter)
            print("Loss:",total_loss,"Train ACC:",acc,"Valid ACC:",v_acc)
            if early_stopping_counter==args.patience:
                print("Early Stopped!")
                break

    #inference mode
    if args.mode=="infer":
        model=HealthModel(args)
        model.load_state_dict(torch.load(args.model_dir+args.goal+args.model_name))
        model.eval()
        test_labels=[]
        test_preds=[]
        for step, batch in enumerate(test_loader):
            cont,cate,label=batch
            cont.to(args.device)
            cate.to(args.device)
            preds=model(cont,cate)

            test_labels.extend(label)
            test_preds.extend(preds.detach())

        if args.goal =="age":
            test_preds=[np.argmax(x) for x in test_preds]
        else:
            test_preds=[1 if x >=0.5 else 0 for x in test_preds]
        pd.DataFrame(test_preds).to_csv(args.output_dir+args.goal+".csv")
        t_acc= accuracy_score(torch.cat(test_labels),test_preds)

        print("Test ACC:",t_acc)



            



    


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
