import pandas as pd
from parser_sensor import parse


def get_raw_data(points, df):
    data_list = []
    label = 0
    i = 0
    for point in points:
        if i < len(df):
            if df.loc[i, "timestamp"] - point.imu.timestamp < 0.5:
                if df.loc[i, "position"] == "手里":
                    label = 1
                else:
                    label = 0
                i += 1
        data = {
            "label": label,
            "ts": point.imu.timestamp,
            "light": point.light.illuminance,
            "gravity_x": point.imu.gravity.x,
            "gravity_y": point.imu.gravity.y,
            "gravity_z": point.imu.gravity.z,
            "magneto_x": point.imu.magneto.x,
            "magneto_y": point.imu.magneto.y,
            "magneto_z": point.imu.magneto.z,
            "magneto_bias_x": point.imu.magneto_bias.x,
            "magneto_bias_y": point.imu.magneto_bias.y,
            "magneto_bias_z": point.imu.magneto_bias.z,
            "net_accel_x": point.imu.net_accel.x,
            "net_accel_y": point.imu.net_accel.y,
            "net_accel_z": point.imu.net_accel.z,
            "raw_accel_x": point.imu.raw_accel.x,
            "raw_accel_y": point.imu.raw_accel.y,
            "raw_accel_z": point.imu.raw_accel.z,
            "raw_accel_bias_x": point.imu.raw_accel_bias.x,
            "raw_accel_bias_y": point.imu.raw_accel_bias.y,
            "raw_accel_bias_z": point.imu.raw_accel_bias.z,
            "gyro_x": point.imu.gyro.x,
            "gyro_y": point.imu.gyro.y,
            "gyro_z": point.imu.gyro.z,
            "gyro_bias_x": point.imu.gyro_bias.x,
            "gyro_bias_y": point.imu.gyro_bias.y,
            "gyro_bias_z": point.imu.gyro_bias.z,
            "quaternion_w": point.imu.quaternion.w,
            "quaternion_x": point.imu.quaternion.x,
            "quaternion_y": point.imu.quaternion.y,
            "quaternion_z": point.imu.quaternion.z,
        }
        data_list.append(data)
    return data_list


fn = "/Users/su/data/phone_play/1609405603-HUAWEI-COL-AL10-10.pb"
action_fn = "/Users/su/data/phone_play/1609405603-HUAWEI-COL-AL10-10-action.csv"

df = pd.read_csv(action_fn)
points = parse(fn, True)
df_raw = pd.DataFrame(get_raw_data(points, df))
print(df_raw)
df_raw.to_csv("data/raw.csv", index=False)
