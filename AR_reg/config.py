class Config(object):
    def __init__(self):
        # model params
        self.model = "AR_reg"
        self.nsteps = 7       # equivalent to x_len
        self.hidden_size = 8
        self.num_heads = 8
        self.gru_size = 8
        self.attention_size = 8
        self.l2_lambda = 1e-3
        self.ar_lambda = 1e-1
        self.ar_g = 0.0

        # data params
        self.data_path = './data/energy_scaled_data_.csv'
        self.nfeatures = 6
        self.x_len = self.nsteps
        self.y_len = 1
        self.foresight = 0
        self.dev_ratio = 0.1
        self.test_len = 15 #1236
        self.seed = None

        #for LSTNet model
        self.kernel_sizes = [3,4,5]
        self.num_filters = 8

        # train params
        self.lr = 1e-3
        self.num_epochs = 100
        self.batch_size = 32
        self.dropout = 0.8
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
