import argparse
import datetime
import json

from google.protobuf.internal.decoder import _DecodeVarint32

import sensor_pb2

last_ts = None


def parse(fn, show_path):
    global last_ts
    points = []

    with open(fn, "rb") as f_pb:
        buffer = f_pb.read()
        start_pos = 0
        while start_pos < len(buffer):
            msg_len, new_pos = _DecodeVarint32(buffer, start_pos)
            msg_buf = buffer[new_pos : new_pos + msg_len]
            start_pos = new_pos + msg_len

            message = sensor_pb2.SensorData()
            message.ParseFromString(msg_buf)

            gnss_ts = message.gnss.timestamp

            last_ts = gnss_ts

            imu_ts = message.imu.timestamp

            t = datetime.datetime.fromtimestamp(gnss_ts)

            points.append(message)
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser("解析传感器pb文件")
    parser.add_argument("-fn", help="pb 文件地址", required=True)
    parser.add_argument(
        "-path", help="0不展示路径, 其余数字展示路径", required=False, default=1, type=int
    )
    args = parser.parse_args()
    points = parse(args.fn, args.path != 0)
    print(points)
