# coding: UTF-8
import time
from datetime import timedelta


def get_time_dif(start_time):
    """
    获取时间间隔
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
