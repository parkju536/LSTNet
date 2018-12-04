class Config(object):
    def __init__(self):
        # data params
        self.data_path = '../data/data.csv'
        self.nfeatures = 8
        self.x_len = 10
        self.y_len = 1
        self.foresight = 0
        self.dev_ratio = 0.1
        self.test_ratio = 0.1
        self.seed = None

        # model params
        self.model = "LSTNet"
        self.attention_size = 128
        self.num_filters = 128
        self.kernel_sizes = [3, 4, 5]
        self.l2_lambda = 1e-3
        self.ar_lambda = 1e-1
        self.ar_g = 1

        # train params
        self.lr = 1e-3
        self.num_epochs = 50
        self.batch_size = 256
        self.nepoch_no_improv = 5
        self.clip = 5
        self.desc = self._desc()

    def _desc(self):
        desc = ""
        for mem, val in self.__dict__.items():
            desc += mem + ":" + str(val) + ", "
        return desc


if __name__ == "__main__":
    config = Config()
    print(config.desc)
