class Config(object):
    def __init__(self):
        # model params
        self.model = "AR_mem"
        self.nsteps = 3        # equivalent to x_len
        self.msteps = 5
        self.attention_size = 32
        self.l2_lambda = 1e-3
        self.ar_lambda = 1e-1
        self.ar_g = 1

        # data params
        self.data_path = './data/MS_scaled_data.csv'
        self.nfeatures = 6
        self.x_len = self.nsteps
        self.y_len = 1
        self.foresight = 0
        self.dev_ratio = 0.1
        self.test_len = 1236
        self.mem_len = self.msteps
        self.seed = None

        # train params
        self.lr = 1e-3
        self.num_epochs = 70
        self.batch_size = 16
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
