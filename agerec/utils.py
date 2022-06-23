from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

import numpy as np
def get_dataloader(train,valid,test,args):
    train_loader, valid_loader,test_loader = None, None ,None
    train_loader = DataLoader(
            train,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size
        )

    valid_loader = DataLoader(
            valid,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size
        )

    test_loader = DataLoader(
            test,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size
        )
    return train_loader,valid_loader,test_loader

def get_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    optimizer.zero_grad()

    return optimizer


