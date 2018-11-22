class Config(object):
    def __init__(self):
        self.lr = 1e-3
        self.attention_size = 128
        self.num_filters = 128
        self.kernel_sizes = [3, 4, 5]
        self.clip = 5
