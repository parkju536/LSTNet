class Config(object):
    def __init__(self):
        # model params
        self.model = "AR"
        self.nsteps = 10        # equivalent to x_len
        self.msteps = 7
        self.nbins = 4
        self.attention_size = 16
        self.num_filters = 32
        self.kernel_sizes = [3]
        self.l2_lambda = 1e-3

        self.hidden_units = 512
        self.num_heads = 8

        # data params
        self.data_path = '../data/18579_12_2mins.csv'
        self.nfeatures = 8
        self.x_len = self.nsteps
        self.y_len = 1
        self.mem_len = self.msteps
        self.foresight = 0
        self.dev_ratio = 0.1
        self.test_len = 7
        self.seed = None

        # train params
        self.lr = 1e-3
        self.num_epochs = 200
        self.batch_size = 32
        self.dropout = 0.9
        self.nepoch_no_improv = 5
        self.clip = 5
        self.desc = self._desc()
        self.allow_gpu = True

    def _desc(self):
        desc = ""
        for mem, val in self.__dict__.items():
            desc += mem + ":" + str(val) + ", "
        return desc


if __name__ == "__main__":
    config = Config()
    print(config.desc)
