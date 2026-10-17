# Adapted from cehrgpt.gpt_utils (same project).
"""
The conventions of the tokens of a patient sequence: which tokens are artificial time tokens, how long they
last, and how ages and times are rebuilt from them.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

import numpy as np
from meds import death_code

from meds_to_cehr.constants import DEATH_TOKEN

logger = logging.getLogger(__name__)


# ETHOS/CoMET time-interval tokens, produced by cehrbert_data's ethos_time_tokens_func, which
# mirrors ethos-ares' `time_intervals_spec`. Between-visit gaps use the bare label
# (e.g. "2mt-6mt"); inpatient (within-visit) gaps prefix it with "i-" (e.g. "i-2mt-6mt") and
# share the label's duration. Gaps of 180 days or more are written as "=6mt" repeated
# round(gap / 180 days) times, so summing per-token durations stays consistent.
#
# Like ethos-ares at inference time (get_token_time), each token is converted back to time
# with the mean duration estimated from the training data (interval_estimates.json from
# ethos-ares' IntervalEstimator, omop/all), in days.
ETHOS_TIME_BUCKET_TO_DAYS = {
    "5m-15m": 0.006483,
    "15m-45m": 0.018534,
    "45m-1h15m": 0.040307,
    "1h15m-2h": 0.067193,
    "2h-3h": 0.103266,
    "3h-5h": 0.163644,
    "5h-8h": 0.245683,
    "8h-12h": 0.416427,
    "12h-18h": 0.606374,
    "18h-1d": 0.832514,
    "1d-2d": 1.299552,
    "2d-4d": 2.855607,
    "4d-7d": 5.413992,
    "7d-12d": 8.927872,
    "12d-20d": 15.233339,
    "20d-30d": 24.212421,
    "30d-2mt": 42.535764,
    "2mt-6mt": 105.303134,
    "=6mt": 180.187472,
}


# Labels of the earlier 13-bucket ETHOS scheme (no overlap with the labels above except
# "5m-15m", which the table above takes over). Kept so datasets and models tokenized with
# that scheme keep parsing.
LEGACY_ETHOS_TIME_BUCKET_TO_DAYS = {
    "15m-1h": 0,
    "1h-2h": 0,
    "2h-6h": 0,
    "6h-12h": 0,
    "12h-1d": 1,
    "1d-3d": 2,
    "3d-1w": 5,
    "1w-2w": 10,
    "2w-1mt": 22,
    "1mt-3mt": 60,
    "3mt-6mt": 135,
    ">=6mt": 180,
}


ALL_ETHOS_TIME_BUCKET_TO_DAYS = {
    **LEGACY_ETHOS_TIME_BUCKET_TO_DAYS,
    **ETHOS_TIME_BUCKET_TO_DAYS,
}


# Regular expression pattern to match inpatient attendance tokens
MEDS_CODE_PATTERN = re.compile(r".*/.*")


INPATIENT_ATT_PATTERN = re.compile(r"(?:VS-|i-)D(\d+)(?:-VE)?")


def is_ethos_time_bucket_token(token: str) -> bool:
    """
    Check if the token is an ETHOS/CoMET time-interval bucket token.

    :param token: Token to check, e.g. "3mt-6mt" or "i-3mt-6mt".
    :return: True if the (optionally "i-"-prefixed) token is a known ETHOS time bucket.
    """
    bucket = token[2:] if token.startswith("i-") else token
    return bucket in ALL_ETHOS_TIME_BUCKET_TO_DAYS


def construct_time_sequence(
    concept_ids: List[str], epoch_times: Optional[List[Union[int, float]]] = None
) -> List[float]:
    if epoch_times is not None:
        return epoch_times

    if concept_ids[0].lower().startswith("year"):
        year_str = concept_ids[0].split(":")[1]
    else:
        year_str = "1985"

    datetime_cursor = datetime(
        int(year_str), month=1, day=1, hour=0, minute=0, second=0
    ).replace(tzinfo=timezone.utc)
    epoch_times = []
    for concept_id in concept_ids:
        if is_att_token(concept_id):
            att_days = extract_time_interval_in_days(concept_id)
            datetime_cursor += timedelta(days=att_days)
        epoch_times.append(datetime_cursor.timestamp())
    return epoch_times


def construct_age_sequence(
    concept_ids: List[str], ages: Optional[List[int]] = None
) -> List[int]:
    if ages is not None:
        return ages
    elif concept_ids[1].lower().startswith("age"):
        age_str = concept_ids[1].split(":")[1]
        try:
            age = max(int(age_str), 0)
        except ValueError:
            logger.warning(
                "Age is not an integer: %s. Setting it to default value 0", age_str
            )
            age = 0
        ages = []
        time_delta = 0
        for concept_id in concept_ids:
            if is_att_token(concept_id):
                time_delta += extract_time_interval_in_days(concept_id)
            ages.append(age + time_delta // 365)
        return ages
    else:
        logger.warning(
            "The second token is not a valid age token. The first 4 tokens are: %s. "
            "Trying to fall back to ages, but it is not valid either %s. "
            "Fall back to a zero vector [0, 0, 0, ...., 0]",
            concept_ids[:4],
            ages,
        )
        return np.zeros_like(concept_ids, dtype=int).tolist()


def multiple_of_10(n: int) -> int:
    return ((n // 10) + 1) * 10


def encode_demographics(
    age: int, gender: int, race: int, max_age=200, max_gender=10, max_race=10
) -> int:
    assert 0 <= age < max_age, f"age: {age}"
    assert 0 <= gender < max_gender, f"gender: {gender}"
    assert 0 <= race < max_race, f"race: {race}"
    return age + max_age * gender + max_age * max_gender * race


def is_clinical_event(token: str, meds: bool = False) -> bool:
    if token.isnumeric():
        return True
    if token in [DEATH_TOKEN, death_code]:
        return True
    if meds:
        return bool(MEDS_CODE_PATTERN.match(token))
    return False


def is_visit_start(token: str):
    """
    Check if the token indicates the start of a visit.

    :param token: Token to check.
    :return: True if the token is a visit start token, False otherwise.
    """
    return token in ["VS", "[VS]"]


def is_visit_end(token: str) -> bool:
    return token in ["VE", "[VE]"]


def is_inpatient_hour_token(token: str) -> bool:
    return token.startswith("i-H")


def extract_time_interval_in_hours(token: str) -> int:
    try:
        hour = int(token[3:])
        return hour
    except ValueError:
        return 0


def is_att_token(token: str):
    """
    Check if the token is an attention token.

    :param token: Token to check.
    :return: True if the token is an attention token, False otherwise.
    """
    if bool(re.match(r"^D\d+", token)):  # day tokens
        return True
    elif bool(re.match(r"^W\d+", token)):  # week tokens
        return True
    elif bool(re.match(r"^M\d+", token)):  # month tokens
        return True
    elif bool(re.match(r"^Y\d+", token)):  # year tokens
        return True
    elif token == "LT":
        return True
    elif token[:3] == "VS-":  # VS-D7-VE
        return True
    elif token[:2] == "i-" and not token.startswith(
        "i-H"
    ):  # i-D7 and exclude hour tokens
        return True
    elif is_ethos_time_bucket_token(token):  # e.g. "3mt-6mt" / "i-3mt-6mt"
        return True
    return False


def is_inpatient_att_token(token: str):
    """
    Check if the token is an inpatient ATT token.

    :param token: Token to check.
    :return: True if the token is an inpatient ATT token, False otherwise.
    """
    if INPATIENT_ATT_PATTERN.match(token):
        return True
    # e.g. "i-3mt-6mt" (ETHOS/CoMET inpatient time bucket, not covered by INPATIENT_ATT_PATTERN)
    return token.startswith("i-") and is_ethos_time_bucket_token(token)


def extract_time_interval_in_days(token: str):
    """
    Extract the time interval in days from a token.

    :param token: Token to extract from.
    :return: Time interval in days.
    :raises ValueError: If the token is invalid.
    """
    try:
        if token[0] == "D":  # day tokens
            return int(token[1:])
        elif token[0] == "W":  # week tokens
            return int(token[1:]) * 7
        elif token[0] == "M":  # month tokens
            return int(token[1:]) * 30
        elif token[0] == "Y":  # year tokens
            return int(token[1:]) * 365
        elif token == "LT":
            return 365 * 3
        elif is_ethos_time_bucket_token(token):  # "3mt-6mt" / "i-3mt-6mt"
            bucket = token[2:] if token.startswith("i-") else token
            return ALL_ETHOS_TIME_BUCKET_TO_DAYS[bucket]
        elif token[:3] == "VS-":  # VS-D7-VE
            part = token.split("-")[1]
            if part.startswith("LT"):
                return 365 * 3
            return int(part[1:])
        elif token[:2] == "i-":  # i-D7
            part = token.split("-")[1]
            if part.startswith("LT"):
                return 365 * 3
            return int(token.split("-")[1][1:])
    except Exception:
        raise ValueError(f"Invalid time token: {token}")
    raise ValueError(f"Invalid time token: {token}")
