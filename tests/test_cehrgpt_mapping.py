import unittest
from datetime import datetime

import numpy as np

from meds_to_cehr.arguments import DataTrainingArguments
from meds_to_cehr.constants import NONE_BIN, UNKNOWN_BIN
from meds_to_cehr.mapping.cehrgpt import (
    HFCehrGptTokenizationMapping,
    MedToCehrGPTDatasetMapping,
)
from meds_to_cehr.meds.schema_extension import CehrBertPatient, Event, Visit
from meds_to_cehr.tokenizer_protocol import SequenceTokenizer


class FakeTokenizer:
    """A tokenizer that satisfies SequenceTokenizer: the mappings need nothing else of one."""

    def __init__(self, tokens, numeric_concept_ids=("4134120",)):
        self._vocab = {token: i for i, token in enumerate(tokens)}
        self._numeric = set(numeric_concept_ids)
        self._values = {NONE_BIN: 0, UNKNOWN_BIN: 1, "BIN:5": 2}

    @property
    def lab_token_ids(self):
        return []

    @property
    def numeric_concept_ids(self):
        return self._numeric

    def get_vocab(self):
        return self._vocab

    def encode(self, concept_ids, **kwargs):
        return [self._vocab[c] for c in concept_ids]

    def decode(self, concept_token_ids, skip_special_tokens=True, **kwargs):
        inverse = {i: t for t, i in self._vocab.items()}
        return [inverse[i] for i in concept_token_ids]

    def normalize(self, concept_id, unit, concept_value):
        return f"BIN:{int(concept_value * 10)}"

    def encode_value(self, concept_values):
        return [self._values[v] for v in concept_values]


def arguments(**kwargs):
    defaults = dict(
        data_folder="unused",
        dataset_prepared_path="unused",
        att_function_type="day",
        inpatient_att_function_type="day",
        include_auxiliary_token=True,
        include_demographic_prompt=True,
    )
    defaults.update(kwargs)
    return DataTrainingArguments(**defaults)


def patient(visits=None):
    outpatient = Visit(
        visit_type="9202",
        visit_start_datetime=datetime(2024, 4, 14),
        events=[Event(time=datetime(2024, 4, 14), code="320128")],
    )
    inpatient = Visit(
        visit_type="9201",
        visit_start_datetime=datetime(2024, 4, 21),
        visit_end_datetime=datetime(2024, 4, 22),
        discharge_facility="8536",
        events=[
            Event(time=datetime(2024, 4, 21), code="320128"),
            Event(time=datetime(2024, 4, 22), code="4134120", numeric_value=0.5),
        ],
    )
    return CehrBertPatient(
        patient_id=7,
        birth_datetime=datetime(1980, 4, 14),
        gender="Gender/F",
        race="Race/unknown",
        ethnicity="Ethnicity/unknown",
        # deliberately out of chronological order
        visits=visits or [inpatient, outpatient],
        index_date=datetime(2024, 4, 22),
        age_at_index=44,
        label=1,
    )


EXPECTED = [
    "year:2024", "age:44", "Gender/F", "Race/unknown",
    "[VS]", "9202", "320128", "[VE]",
    "D7",
    "[VS]", "9201", "320128", "i-D1", "4134120", "8536", "[VE]",
]  # fmt: skip


class MedToCehrGPTDatasetMappingTest(unittest.TestCase):
    def setUp(self):
        self.record = MedToCehrGPTDatasetMapping(arguments()).transform(patient())

    def test_visits_are_sorted_in_time_and_framed_by_visit_tokens(self):
        self.assertEqual(self.record["concept_ids"], EXPECTED)

    def test_every_per_token_column_has_one_entry_per_token(self):
        n = len(EXPECTED)
        for column in ("ages", "epoch_times", "concept_value_masks", "units"):
            self.assertEqual(len(self.record[column]), n, column)
        self.assertEqual(self.record["num_of_concepts"], n)
        self.assertEqual(self.record["num_of_visits"], 2)

    def test_times_never_go_back(self):
        times = self.record["epoch_times"]
        self.assertEqual(times, sorted(times))
        # the second visit is 7 days after the first
        self.assertEqual(times[9] - times[4], 7 * 24 * 3600)

    def test_only_the_lab_value_is_flagged(self):
        masks = self.record["concept_value_masks"]
        self.assertEqual([i for i, m in enumerate(masks) if m], [13])
        self.assertEqual(self.record["concept_ids"][13], "4134120")

    def test_the_auxiliary_token_is_the_discharge_facility_of_an_inpatient_visit(self):
        record = MedToCehrGPTDatasetMapping(
            arguments(include_auxiliary_token=False)
        ).transform(patient())
        self.assertEqual(
            record["concept_ids"], [c for c in EXPECTED if c != "8536"]
        )


class HFCehrGptTokenizationMappingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer(EXPECTED + ["Unseen"])
        self.record = MedToCehrGPTDatasetMapping(arguments()).transform(patient())

    def test_the_fake_tokenizer_is_a_sequence_tokenizer(self):
        self.assertIsInstance(self.tokenizer, SequenceTokenizer)

    def test_the_tokens_are_encoded_and_numeric_values_are_binned(self):
        out = HFCehrGptTokenizationMapping(self.tokenizer).transform(self.record)
        self.assertEqual(out["input_ids"], self.tokenizer.encode(EXPECTED))
        values = out["values"]
        # the value 0.5 of the lab falls in the bin the tokenizer gives it, every other token has none
        self.assertEqual(values[13], 2)
        self.assertEqual(sum(1 for v in values if v != 0), 1)

    def test_tokens_that_are_not_in_the_vocabulary_are_dropped_from_every_column(self):
        tokenizer = FakeTokenizer([t for t in EXPECTED if t != "320128"])
        out = HFCehrGptTokenizationMapping(tokenizer).transform(self.record)
        self.assertNotIn("320128", out["concept_ids"])
        n = len(EXPECTED) - 2
        self.assertEqual(len(out["input_ids"]), n)
        for column in ("ages", "epoch_times", "values", "value_indicators"):
            self.assertEqual(len(out[column]), n, column)

    def test_a_value_of_a_concept_without_statistics_is_the_unknown_bin(self):
        tokenizer = FakeTokenizer(EXPECTED, numeric_concept_ids=())
        out = HFCehrGptTokenizationMapping(tokenizer).transform(self.record)
        self.assertEqual(out["values"][13], 1)


class RecordAsArraysTest(unittest.TestCase):
    def test_a_zero_token_is_renamed_so_it_cannot_break_the_tokenizer(self):
        record = MedToCehrGPTDatasetMapping(arguments()).transform(patient())
        record["concept_ids"] = np.array(["0" if c == "320128" else c for c in record["concept_ids"]])
        tokenizer = FakeTokenizer([t for t in EXPECTED if t != "320128"] + ["Unknown"])
        out = HFCehrGptTokenizationMapping(tokenizer).transform(record)
        self.assertEqual(list(out["concept_ids"]).count("Unknown"), 2)
