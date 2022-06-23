import argparse


def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument(
        "--train_dir", default='../data/preprocessed/train_data.csv', type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="../models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="../output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_dir", default="../data/preprocessed/test_data.csv", type=str, help="test file name"
    )

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=512, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0005, type=float, help="learning rate")
    parser.add_argument("--patience", default=10, type=int, help="for early stopping")
    parser.add_argument("--hidden_dim",default=128,type=int,help="embedding dim")
   
    ### 중요 ###
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument("--n_kfold", default=5, type=int, help="numb of folds")
    parser.add_argument("--goal", default='age', type=str, help="age or sex")
    parser.add_argument("--mode", default='train', type=str, help="train or inference")
    args = parser.parse_args()

    return args
