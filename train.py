import argparse
import os
from utils import get_logger, make_date_dir
from data_utils import load_data_from_csv, batch_loader
import numpy as np
from time import time


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("-m", "--model", type=str, default="LSTNet",
                        required=True, choices=["LSTNet", "MANN"],
                        help="Model selected in the list: LSTNet, MANN")

    # Optional parameters

    args = parser.parse_args()
    if args.model == "LSTNet":
        from LSTNet.config import Config
        from LSTNet.model import Model
        config = Config()

    else:
        from MANN.config import Config
        from MANN.model import Model
        config = Config()

    logger = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration======")
    logger.info(config.desc)
    logger.info("================================")

    try:
        train_x, dev_x, test_x, train_y, dev_y, test_y = load_data_from_csv(data_path=config.data_path,
                                                                            x_len=config.x_len,
                                                                            y_len=config.y_len,
                                                                            foresight=config.foresight,
                                                                            dev_ratio=config.dev_ratio,
                                                                            test_ratio=config.test_ratio,
                                                                            seed=config.seed)
        logger.info("train_x shape: {}, dev_x shape: {}, test_x shape: {}"
                    .format(train_x.shape, dev_x.shape, test_x.shape))
        logger.info("train_y shape: {}, dev_y shape: {}, test_y shape: {}"
                    .format(train_y.shape, dev_y.shape, test_y.shape))
        model = Model(config)
        train_data = list(zip(train_x, train_y))
        no_improv = 0
        best_loss = 100
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        result_dir = make_date_dir(os.path.join(config.model, 'results/'))

        start_time = time()
        for i in range(config.num_epochs):
            train_batches = batch_loader(train_data, config.batch_size)
            epoch = i+1

            for batch in train_batches:
                batch_x, batch_y = zip(*batch)
                loss, rse, mape, mae, step = model.train(batch_x, batch_y)

                if step % 100 == 0:
                    logger.info("epoch: {ep}, step: {st}, loss: {lo:.4f}, rse: {rs:.4f}, mape: {map:.4f}, mae: {ma:.4f}".format(ep=epoch, st=step, lo=loss, rs=rse, map=mape, ma=mae))

            # dev score for each epoch (no mini batch)
            _, dev_loss, dev_rse, dev_mape, dev_mae = model.eval(dev_x, dev_y)

            if dev_loss < best_loss:
                best_loss = dev_loss
                no_improv = 0
                logger.info("New score! : dev_loss: {lo:.4f}, dev_rse: {rs:.4f}, dev_mape: {map:.4f}, dev_mae: {ma:.4f}".format(lo=dev_loss, rs=dev_rse, map=dev_mape, ma=dev_mae)
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
        pred, test_loss, test_rse, test_mape, test_mae = model.eval(test_x, test_y)
        logger.info("test_loss: {lo:.4f}, test_rse: {rs:.4f}, test_mape: {map:.4f}, test_mae: {ma:.4f}".format(lo=test_loss, rs=test_rse, map=test_mape, ma=test_mae))

        # save results
        np.save(os.path.join(result_dir, 'pred.npy'), pred)
        np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        logger.info("Saving results at {}".format(result_dir))
        logger.info("Elapsed training time {0:0.4f}".format(elapsed))
        logger.info("Training finished, exit program")

    except:
        logger.exception("ERROR")


if __name__ == "__main__":
    main()
