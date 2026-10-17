"""Helpers of the tests: tiny MEDS datasets as meds_reader databases."""

import datetime as dt
import json
import os
import shutil
import subprocess
import sys

import polars as pl

from meds_to_cehr.arguments import DataTrainingArguments


def converter():
    """The meds_reader_convert script of the environment, None if not installed."""
    candidates = (
        os.path.join(os.path.dirname(sys.executable), "meds_reader_convert"),
        shutil.which("meds_reader_convert"),
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_database(root, name, codes):
    """Three subjects (train, tuning, held_out) with the given codes, as a meds_reader database."""
    source = os.path.join(root, name)
    os.makedirs(os.path.join(source, "data"))
    os.makedirs(os.path.join(source, "metadata"))
    subjects = (1, 2, 3)
    rows = [
        (subject, dt.datetime(2020, 1, 1 + i), code, None)
        for subject in subjects
        for i, code in enumerate(codes)
    ]
    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.String,
        "numeric_value": pl.Float32,
    }
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(
        os.path.join(source, "data", "0.parquet")
    )
    pl.DataFrame(
        {"subject_id": list(subjects), "split": ["train", "tuning", "held_out"]}
    ).write_parquet(os.path.join(source, "metadata", "subject_splits.parquet"))
    with open(os.path.join(source, "metadata", "dataset.json"), "w") as f:
        json.dump(
            {
                "dataset_name": name,
                "dataset_version": "1",
                "etl_name": "t",
                "etl_version": "1",
                "meds_version": "0.4.0",
            },
            f,
        )
    pl.DataFrame(
        {"code": sorted(set(codes)), "description": None, "parent_codes": None},
        schema={
            "code": pl.String,
            "description": pl.String,
            "parent_codes": pl.List(pl.String),
        },
    ).write_parquet(os.path.join(source, "metadata", "codes.parquet"))
    database = os.path.join(root, name + "_db")
    subprocess.run(
        [converter(), source, database, "--num_threads", "1"],
        check=True,
        capture_output=True,
    )
    return database


def data_args(folder, conversion_type=None, **kwargs):
    arguments = dict(
        data_folder=folder,
        dataset_prepared_path=os.path.join(folder, "prepared"),
        is_data_in_meds=True,
        preprocessing_num_workers=1,
    )
    if conversion_type:
        arguments["meds_to_cehrbert_conversion_type"] = conversion_type
    arguments.update(kwargs)
    return DataTrainingArguments(**arguments)
