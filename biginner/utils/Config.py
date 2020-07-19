
class Config:
    use_gpu = True

    num_class = 10
    data_root = ''

    # the parameters need to change while training
    epoch = 5
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 5e-4


config = Config()

