import sklearn


class EML(object):
    def __init__(
        self,
        train_df,
        test_df,
        model,
        feature_func=lambda df: df,
        label_column=None,
        feature_columns=None,
    ):
        super(EML).__init__()
        self.tag = ["init"]
        self.train_dfs = [train_df]
        self.test_dfs = [test_df]
        self.label_column = label_column if label_column else "label"
        self.feature_columns = (
            feature_columns
            if feature_columns
            else [x for x in train_df.columns if x != "label"]
        )
        self.feature_func = feature_func

    def feature_engineer(self):
        self.train_df = self.feature_func(self.train_df)
        self.test_df = self.feature_func(self.test_df)
        self.train_df_ = self.train_df.copy()
        return self.train_df.columns.to_list()

    def set_columns(self, label_column, feature_columns):
        self.feature_columns = feature_columns
        self.label_column = label_column

    def outlier_removing(self, pipeline):
        pass

    def standarding(self, pipeline):
        if pipeline == None:
            pipeline = self.processing
        train = pipeline.fit(self.train_df[self.feature_columns])
        test = pipeline.transform(self.train_df[self.feature_columns])
        return train, test

    def train_model(self, model=None):
        label_train, feature_train, label_test, feature_test = self.get_data_for_model()
        # pipeline  self.

    def get_current_df(self):
        return self.train_dfs[-1], self.test_dfs[-1]

    def get_data_for_model(self):
        train_df, test_df = self.get_current_df()
        return (
            self.train_df[self.label_column],
            self.train_df[self.feature_columns],
            self.test_df[self.label_column],
            self.test_df[self.feature_columns],
        )

    def evaluate(self):
        pass

    def pretify(self):
        pass


if __name__ == "__main__":
    e = EML()