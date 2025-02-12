#!/usr/bin/env python
# coding: utf-8
import logging

from run.run import run_train, run_test
from run.run2 import ensemble_run_train, ensemble_run_test
from run.tuning import tuning_start
from tools import log, parser

logger = logging.getLogger(__name__)

def main():
    # 引数の設定・読み込み
    args = parser.get_argument()

    # ファイルの保存先を作成
    #experiment_path, labeldir_path = misc.make_experiment_dir(args)

    # ログの設定
    log.init_log_setting(args)

    # 実験の設定を出力
    log.print_log_setting(args)

    # シード値の固定
    #misc.fix_seed(args) 

    if args.test:
        run_test(args)
    else:
        run_train(args)
        run_test(args)
        #ensemble_run_train(args)
        #ensemble_run_test(args)
        #tuning_start(args)

    logger.info('finish')
        

if __name__ == '__main__':
    main()