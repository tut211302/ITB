import argparse

def get_argument():
    """引数の設定を行う

    Returns:
        _type_: パラメータを含む引数
    """

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself ! Cool huh ? :D')
    parser.add_argument('--data_dir', type = str, default = '/home/fukuyama/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3', help = 'This is the path of the training data')
    parser.add_argument('--bs', type = int, default = 64, help = 'batch size')
    parser.add_argument('--epochs',           type=int,   default=3)
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate for the optimizer')
    #parser.add_argument('--stage', type = int, default = 1, help = 'Stage, it decides which layers of the Neural Net to train')
    parser.add_argument('-pc, --train_pc', type = float, default = 0.8)
    parser.add_argument('--loss_func', type = str, default = 'BCE', choices = {'BCE'}, help = 'loss function')
    #parser.add_argument('-r','--resume', action = 'store_true') # args.resume will return True if -r or --resume is used in the terminal
    parser.add_argument('--ckpt', type = str, default = 'best.pth', help = 'Path of the ckeckpoint that you want to load')
    parser.add_argument('-t','--test', action = 'store_true')   # args.test   will return True if -t or --test   is used in the terminal

    parser.add_argument('--pkl_dir_path'  ,       default = 'pickles')
    parser.add_argument('--train_val_df_pkl_path', default = 'train_val_df.pickle')
    parser.add_argument('--test_df_pkl_path',       default = 'test_df.pickle')
    parser.add_argument('--disease_classes_pkl_path', default = 'disease_classes.pickle')
    parser.add_argument('--models_dir',               default = 'models')

    parser.add_argument('--threshold',         type=float,   default=0.5)
    #parser.add_argument('--experiment_path',    type=str, default='/home/fukuyama/ITB/experiment')
    parser.add_argument('--experiment_name',    type=str, default='sample')
    parser.add_argument('--model_name',         type=str, default = 'densenet121-res224-chex')
    parser.add_argument('--disease_name',         type=str, default = 'Effusion')
    parser.add_argument('--class_numbers',         type=int, default = '15')
    #parser.add_argument('--model_state',      type=str,   default='best.pth')

    args = parser.parse_args()

    args.experiment_path = f'/home/fukuyama/ITB/experiment/{args.class_numbers}/{args.model_name}/'

    print(f"Experiment path: {args.experiment_path}")

    return args