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
from .quaternion import Quaternion
import numpy as np


def get_circle_mean(partial_series):
    return np.angle(np.exp(partial_series * 1j).sum())


def get_mathematical_mean(partial_series):
    return np.mean(partial_series)


def get_circle_diff(partial_series, local_center):
    diff = partial_series.apply(
        lambda x: x
        - local_center
        - 2 * np.pi * ((x - local_center + np.pi) // (2 * np.pi))
    )
    return diff


def get_mathematical_diff(partial_series, local_center):
    return partial_series - local_center


def partial_norm(df_euler, start_idx, end_idx, yaw_idx, method="circle"):
    df_euler_ = df_euler.copy()
    mean_fun = {"circle": get_circle_mean, "mathematical": get_mathematical_mean}[
        method
    ]
    diff_fun = {"circle": get_circle_diff, "mathematical": get_mathematical_diff}[
        method
    ]
    local_center = mean_fun(df_euler_.iloc[start_idx:end_idx, yaw_idx])
    df_euler_.iloc[start_idx:end_idx, yaw_idx] = diff_fun(
        df_euler_.iloc[start_idx:end_idx, yaw_idx], local_center
    )
    return df_euler_


def add_euler_angle(
    df, w_column, x_column, y_column, z_column, yaw_norm="glocal-mathematical"
):
    df_euler = df.copy()
    df_euler["roll"] = 0
    df_euler["pitch"] = 0
    df_euler["yaw"] = 0
    for i in df_euler.index:
        w = df_euler.loc[i, w_column]
        x = df_euler.loc[i, x_column]
        y = df_euler.loc[i, y_column]
        z = df_euler.loc[i, z_column]
        roll, pitch, yaw = get_ENU_euler_angle(w, x, y, z)
        df_euler.loc[i, "roll"] = roll
        df_euler.loc[i, "pitch"] = pitch
        df_euler.loc[i, "yaw"] = yaw

    yaw_idx = list(df_euler.columns).index("yaw")
    if yaw_norm.split("-")[0] == "global":
        df_euler = partial_norm(
            df_euler,
            0,
            len(df_euler),
            yaw_idx,
            method=yaw_norm.split("-")[1],
        )
    elif yaw_norm.split("-")[0] == "local":
        local_window_size = 123
        for i in range(len(df_euler) // local_window_size):
            df_euler = partial_norm(
                df_euler,
                i * local_window_size,
                i * local_window_size + local_window_size,
                yaw_idx,
                method=yaw_norm.split("-")[1],
            )
        if i * local_window_size + local_window_size < len(df_euler):
            df_euler = partial_norm(
                df_euler,
                i * local_window_size + local_window_size,
                len(df_euler),
                yaw_idx,
                method=yaw_norm.split("-")[1],
            )

    return df_euler


def get_ENU_euler_angle(w, x, y, z, os_type="Android"):
    q = Quaternion(w=w, x=x, y=y, z=z)
    euler_angle = q.to_rpy()
    return euler_angle  # East North Up: most use


def get_axisdata_distance(df, axis_feature_list):
    df_distance = df.copy()
    for feature in axis_feature_list:
        df_distance[feature + "_hypt"] = (
            df_distance[feature + "_x"] ** 2
            + df_distance[feature + "_y"] ** 2
            + df_distance[feature + "_z"] ** 2
        ) ** 0.5
    return df_distance


def get_window_data(df, half_window_len):
    data_list = []
    for i in range(len(df)):
        data = get_window_data_one_row(df, i, half_window_len)
        if data:
            data_list.append(data)
    return pd.DataFrame(data_list)


def get_window_data_one_row(df, idx, half_window_len):
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


def calc_window_feature():
    pass


if __name__ == "__main__":
    df = pd.read_csv("data/raw.csv")

    df_window = get_window_data(df, 10)

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
