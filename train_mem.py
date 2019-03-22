import argparse
import os
from utils import get_logger, make_date_dir
from data_utils import load_data_from_csv, load_data_mem, batch_loader
import matplotlib.pyplot as plt
import numpy as np
from time import time


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("-m", "--model", type=str, default="LSTNet",
                        required=True, choices=["LSTNet", "MANN", "AR_reg", "AR", "AR_mem"],
                        help="Model selected in the list: LSTNet, MANN, AR_reg, AR, AR_mem")

    # Optional parameters

    args = parser.parse_args()
    if args.model == "LSTNet":
        from LSTNet.config import Config
        from LSTNet.model import Model
        config = Config()
    elif args.model == "MANN":
        from MANN.config import Config
        from MANN.model import Model
        config = Config()
    elif args.model == "AR_reg":
        from AR_reg.config import Config
        from AR_reg.model import Model
        config = Config()
    elif args.model == "AR_mem":
        from AR_mem.config import Config
        from AR_mem.model import Model
        config = Config()
    elif args.model == "AR":
        from AR.config import Config
        from AR.model import Model
        config = Config()

    COL_LIST = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    logger = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration======")
    logger.info(config.desc)
    logger.info("================================")

    try:
                                                                    
        train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = load_data_mem(data_path=config.data_path,
                                                                            x_col_list = COL_LIST,
                                                                            y_col_list = COL_LIST,
                                                                            x_len=config.x_len,
                                                                            y_len=config.y_len,
                                                                            mem_len = config.mem_len,
                                                                            foresight=config.foresight,
                                                                            dev_ratio=config.dev_ratio,
                                                                            test_len = config.test_len,
                                                                            seed=config.seed)
                                                                            
        model = Model(config)
        train_data = list(zip(train_x, train_m, train_y))
        no_improv = 0
        best_loss = 100
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        result_dir = make_date_dir(os.path.join(config.model, 'results/'))
        logger.info("Start training")
        dev_x = np.asarray(dev_x)
        dev_y = np.asarray(dev_y)

        start_time = time()
        for i in range(config.num_epochs):
            train_batches = batch_loader(train_data, config.batch_size)
            epoch = i+1

            for batch in train_batches:
                batch_x, batch_m, batch_y = zip(*batch)
                loss, rmse, rse, smape, mae, step = model.train(batch_x, batch_m, batch_y)

                if step % 100 == 0:
                    logger.info("epoch: %d, step: %d, loss: %4f, rmse: %4f, rse: %4f, smape: %4f, mae: %4f" %
                                (epoch, step, loss, rmse, rse, smape, mae))

            # dev score for each epoch (no mini batch)
            _, dev_loss, dev_rmse, dev_rse, dev_smape, dev_mae = model.eval(dev_x, dev_m, dev_y)

            if dev_loss < best_loss:
                best_loss = dev_loss
                no_improv = 0
                logger.info("New score! : dev_loss: %4f, rmse: %4f, dev_rse: %4f, dev_smape: %4f, dev_mae: %4f" %
                            (dev_loss, dev_rmse, dev_rse, dev_smape, dev_mae))
                logger.info("Saving model at {}".format(model_dir))
                model.save_session(os.path.join(model_dir, config.model))
            else:
                no_improv += 1
                if no_improv == config.nepoch_no_improv:
                    logger.info("No improvement for %d epochs" % no_improv)
                    break

        elapsed = time()-start_time
        # generating results (no mini batch)
        model.restore_session(model_dir)
        pred, test_loss, test_rmse, test_rse, test_smape, test_mae = model.eval(test_x, test_m, test_y)
        logger.info("test_loss: %4f, test_rmse: %4f, test_rse: %4f, test_smape: %4f, test_mae: %4f" %
                    (test_loss, test_rmse, test_rse, test_smape, test_mae))

        # save results
        np.save(os.path.join(result_dir, 'pred.npy'), pred)
        np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        np.save(os.path.join(result_dir, 'test_dt.npy'), test_y)
        logger.info("Saving results at {}".format(result_dir))
        logger.info("Elapsed training time {0:0.2f}".format(elapsed/60))
        logger.info("Training finished, exit program")

        t = np.linspace(0, pred.shape[0], num=pred.shape[0])
        mae = np.mean(np.abs(test_y-pred))
        mape = np.mean(np.abs((test_y-pred)/test_y))
        plt.rcParams['figure.figsize']=[20,4]
        plt.plot(t, test_y, "r", alpha=0.5)
        #plt.ylim(0.5,1.0)
        plt.plot(t,pred,"b")
        #plt.title("{}, mape:{mape:.5f}, mae:{mae:.5f}".format(raw.columns[1], mape=mape, mae=mae), size=20)
        plt.legend(("actual","pred"),loc="upper left")
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(config.model, "image/figure.png"))

    except:
        logger.exception("ERROR")


if __name__ == "__main__":
    main()
