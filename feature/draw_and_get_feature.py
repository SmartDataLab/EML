import matplotlib.pyplot as plt
from parser_sensor import parse
import pandas as pd

fn = "/Users/su/data/phone_play/1609405603-HUAWEI-COL-AL10-10.pb"
action_fn = "/Users/su/data/phone_play/1609405603-HUAWEI-COL-AL10-10-action.csv"

df = pd.read_csv(action_fn)
print(df)

points = parse(fn, True)
start_ts = points[0][0]
print(start_ts)
ts_list = [x[0] for x in points]
print((max(ts_list) - min(ts_list)) / len(ts_list))
illuminance = [x[-3].gravity.x for x in points]
plt.plot(illuminance)
for ts in df[df["position"] == "手里"]["timestamp"]:
    plt.vlines((ts - start_ts) * 10, min(illuminance), max(illuminance), color="r")
for ts in df[df["position"] == "口袋里"]["timestamp"]:
    plt.vlines((ts - start_ts) * 10, min(illuminance), max(illuminance), color="b")
plt.show()