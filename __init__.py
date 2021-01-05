import sklearn

def download_dataset():

class EML(object):
    def __init__(self, train_df, test_df, model, feature_func=lambda df: df, label_column =None, feature_columns= None):
        super(EML).__init__()
        self.train_df = trian_df
        self.test_df = test_df
        self.label_column = label_column if label_column else 'label'
        self.feature_columns = feature_columns if feature_columns else [x for x in train_df.columns if x != 'label']
        self.model = model
        self.feature_func = feature_func
        self.result = None

    def feature_engineer(self):
        self.train_df = self.feature_func(self.train_df)
        self.test_df = self.feature_func(self.test_df)
        return self.train_df.columns.to_list()

    def set_columns(self, label_column, feature_columns):
        self.feature_columns == feature_columns
        self.label_column = label_column

    def train_model(self, pipeline):
        label_train, feature_train = self.train_df[self.label_column]], self.train_df[self.feature_columns]
        label_test, feature_test = self.test_df[self.label_column], self.test_df[self.feature_columns]
        pipeline
        

    def evaluate(self):
        pass

    def pretify(self):
        pass

if __name__ == '__main__':
    e = EML()