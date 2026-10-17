# Adapted from cehrbert_data.decorators.patient_event_decorator_base (MIT License, same copyright holder as cehrgpt).
"""The artificial time tokens (ATT) between visits and the functions that create them."""

import math
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np


class AttType(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    CEHR_BERT = "cehr_bert"
    MIX = "mix"
    NONE = "none"


def time_token_func(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 0:
        return "W-1"
    if time_delta < 28:
        return f"W{str(math.floor(time_delta / 7))}"
    if time_delta < 360:
        return f"M{str(math.floor(time_delta / 30))}"
    return "LT"


def time_day_token(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"D{str(time_delta)}"
    return "LT"


def time_week_token(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"W{str(math.floor(time_delta / 7))}"
    return "LT"


def time_month_token(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"M{str(math.floor(time_delta / 30))}"
    return "LT"


def time_mix_token(time_delta: int) -> Optional[str]:
    #        WHEN day_diff <= 7 THEN CONCAT('D', day_diff)
    #         WHEN day_diff <= 30 THEN CONCAT('W', ceil(day_diff / 7))
    #         WHEN day_diff <= 360 THEN CONCAT('M', ceil(day_diff / 30))
    #         WHEN day_diff <= 720 THEN CONCAT('Q', ceil(day_diff / 90))
    #         WHEN day_diff <= 1440 THEN CONCAT('Y', ceil(day_diff / 360))
    #         ELSE 'LT'
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta <= 7:
        return f"D{str(time_delta)}"
    if time_delta <= 30:
        # e.g. 8 -> W2
        return f"W{str(math.ceil(time_delta / 7))}"
    if time_delta <= 360:
        # e.g. 31 -> M2
        return f"M{str(math.ceil(time_delta / 30))}"
    # if time_delta <= 720:
    #     # e.g. 361 -> Q5
    #     return f'Q{str(math.ceil(time_delta / 90))}'
    # if time_delta <= 1080:
    #     # e.g. 1081 -> Y2
    #     return f'Y{str(math.ceil(time_delta / 360))}'
    return "LT"


def get_att_function(att_type: Union[AttType, str]) -> Callable:
    # Convert the att_type str to the corresponding enum type
    if isinstance(att_type, str):
        att_type = AttType(att_type)

    if att_type == AttType.DAY:
        return time_day_token
    elif att_type == AttType.WEEK:
        return time_week_token
    elif att_type == AttType.MONTH:
        return time_month_token
    elif att_type == AttType.MIX:
        return time_mix_token
    elif att_type == AttType.CEHR_BERT:
        return time_token_func
    return None
