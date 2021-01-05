import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

df = pd.read_csv("data/raw.csv")

print(df["label"].sum())


def get_window_data(df, idx, half_window_len):
    if idx < half_window_len or idx >= len(df) - half_window_len:
        return None
    feature_list = list(df.columns)
    feature_list.pop(feature_list.index("label"))
    feature_list.pop(feature_list.index("ts"))
    data = {"label": df.loc[idx, "label"]}
    for feature in feature_list:
        for i in range(-half_window_len, half_window_len + 1):
            data["%s_%s" % (feature, i)] = df.loc[idx + i, feature]
    return data


data_list = []
for i in range(len(df)):
    data = get_window_data(df, i, 10)
    if data:
        data_list.append(data)

df_window = pd.DataFrame(data_list)

print(df_window)

feature_list = list(df_window.columns)
feature_list.pop(feature_list.index("label"))

target, feature = df_window["label"], df_window[feature_list]
scaler = StandardScaler()
scaler.fit(feature)
feature = scaler.transform(feature)
feature_train, feature_test, target_train, target_test = train_test_split(
    feature, target, test_size=0.3, random_state=0
)

clf = RandomForestClassifier()
clf.fit(feature_train, target_train)
predict_results = clf.predict(feature_test)
conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))

clf2 = Lasso(alpha=0.1)
clf2.fit(feature_train, target_train)
importance_list = tuple(zip(feature_list, clf2.coef_))
sorted_list = sorted(importance_list, key=lambda x: x[1], reverse=True)
print(sorted_list)
predict_results = [1.0 if x > 0.5 else 0.0 for x in clf2.predict(feature_test)]
print(predict_results)
conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))

importance_list = tuple(zip(feature_list, clf.feature_importances_))

sorted_list = sorted(importance_list, key=lambda x: x[1], reverse=True)

plt.bar([i + 1 for i in range(10)], [sorted_list[i][1] for i in range(10)])
plt.xticks(
    [i + 1 for i in range(10)],
    [sorted_list[i][0] for i in range(10)],
    rotation=30,
)
plt.show()
