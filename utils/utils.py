import datetime
import time


def get_time_acc():
    time_str = str(datetime.datetime.now())[:-3].replace(
        ' ', '').replace(":", '').replace('-', '')
    return time_str