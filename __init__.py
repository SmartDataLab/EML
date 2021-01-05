class EML(object):
    def __init__(self, train_df, test_df, model, feature=lambda df: df, label_column =):
        super(EML).__init__()
        self.data_loader = data_loader
        self.model = model
        self.featue = feature

    def preprocessing(self):
        pass

    def feature_engineer(self):
        pass

    def train_model(self):
        pass

    def evaluate(self):
        pass