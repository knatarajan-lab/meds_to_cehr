import os
import shutil
import tempfile
import unittest

from datasets import config as datasets_config
from meds_fixtures import build_database, converter, data_args
from test_cehrgpt_mapping import FakeTokenizer

from meds_to_cehr.mapping.cehrgpt import MedToCehrGPTDatasetMapping
from meds_to_cehr.mapping.cehrgpt_datasets import create_cehrgpt_pretraining_dataset
from meds_to_cehr.meds.conversion_detection import resolve_meds_conversion_type
from meds_to_cehr.meds.meds_utils import create_dataset_from_meds_reader

# a MIMIC-IV style stay: no argument tells the conversion what the data looks like
MIMIC_STAY = [
    "MEDS_BIRTH",
    "Gender/F",
    "Race/White",
    "HOSPITAL_ADMISSION//EW EMER.",
    "LAB//1",
    "HOSPITAL_DISCHARGE//HOME",
]
SEQUENCE = [
    "year:2020", "age:0", "Gender/F", "Race/White",
    "[VS]", "Visit/IP", "HOSPITAL_ADMISSION//EW_EMER.", "i-D1", "LAB//1", "HOSPITAL_DISCHARGE//HOME", "[VE]",
]  # fmt: skip


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        # keep the cache of the datasets library out of the home folder
        cls.old_cache = datasets_config.HF_DATASETS_CACHE
        datasets_config.HF_DATASETS_CACHE = os.path.join(cls.root, "hf_cache")
        cls.database = build_database(cls.root, "mimic", MIMIC_STAY)

    @classmethod
    def tearDownClass(cls):
        datasets_config.HF_DATASETS_CACHE = cls.old_cache
        shutil.rmtree(cls.root, ignore_errors=True)

    def convert(self):
        args = data_args(
            self.database,
            att_function_type="day",
            inpatient_att_function_type="day",
            include_auxiliary_token=True,
        )
        resolved = resolve_meds_conversion_type(args)
        dataset = create_dataset_from_meds_reader(
            args, dataset_mappings=[MedToCehrGPTDatasetMapping(args)]
        )
        return args, resolved, dataset

    def test_a_mimic_dataset_becomes_cehrgpt_sequences_without_naming_its_shape(self):
        _, resolved, dataset = self.convert()
        self.assertEqual(resolved, "MedsToBertMimic4")
        # the three splits of the MEDS dataset
        self.assertEqual({k: v.num_rows for k, v in dataset.items()}, {"train": 1, "validation": 1, "test": 1})
        row = dataset["train"][0]
        # the stay is one inpatient visit, not an admission and a discharge cut apart
        self.assertEqual(row["concept_ids"], SEQUENCE)
        self.assertEqual(row["num_of_visits"], 1)
        self.assertEqual(row["num_of_concepts"], len(SEQUENCE))

    def test_the_sequences_can_be_tokenized(self):
        args, _, dataset = self.convert()
        tokenizer = FakeTokenizer(SEQUENCE)
        tokenized = create_cehrgpt_pretraining_dataset(dataset, tokenizer, args)
        row = tokenized["train"][0]
        self.assertEqual(row["input_ids"], tokenizer.encode(SEQUENCE))
        self.assertEqual(len(row["epoch_times"]), len(SEQUENCE))
