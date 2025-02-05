import os

import logging

logger = logging.getLogger(__name__)

def init_log_setting(args):
    """ログ出力の設定を行う

    * 出力先の設定

    * 出力フォーマットの設定

    Args:
        args (_type_): パラメータの設定された引数
    """
    stream_hd = logging.StreamHandler()

    log_path = f'{args.experiment_path}/{args.model_name}/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if args.test:
        file_hd = logging.FileHandler(os.path.join(f'{log_path}/', f'test.log'))
    else:
        file_hd = logging.FileHandler(os.path.join(f'{log_path}/', f'train.log'))


    logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s] : %(message)s', \
                        level=logging.INFO, handlers=[stream_hd, file_hd])
    
    logger.info('logging setting complete.')

def print_log_setting(args):
    """設定一覧のログへの出力

    Args:
        args (_type_): パラメータの設定された引数
    """

    logger.info('----------------------args setting----------------------')
    
    member_in_args = vars(args)
    for key, value in member_in_args.items():
        logger.info(f'{key.ljust(25, " ")}:  {value}')