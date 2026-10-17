import shutil
import tempfile
import unittest

import meds_reader
from meds_fixtures import build_database, converter, data_args

from meds_to_cehr.meds import conversion_detection as mc
from meds_to_cehr.meds.meds_utils import (
    convert_one_patient,
    get_meds_to_cehrbert_conversion_cls,
)

OMOP = "MedsToCehrbertOMOP"
MIMIC = "MedsToBertMimic4"
OMOP_CODES = ["MEDS_BIRTH", "Visit/IP", "Visit/ER", "SNOMED/201826", "LOINC/1"]
MIMIC_CODES = [
    "MEDS_BIRTH",
    "HOSPITAL_ADMISSION//EW EMER.",
    "HOSPITAL_DISCHARGE//HOME",
    "ED_REGISTRATION//X",
    "LAB//50827//UNK",
]
STAY_CODES = ["MEDS_BIRTH", "HOSPITAL_ADMISSION//EW EMER.", "HOSPITAL_DISCHARGE//HOME"]
LOGGER = "meds_to_cehr.meds.conversion_detection"


class DetectConversionTypeTest(unittest.TestCase):
    def test_both_conversions_are_registered(self):
        self.assertEqual(mc.registered_conversion_types(), [MIMIC, OMOP])

    def test_the_conversion_whose_rules_match_the_codes_is_found(self):
        self.assertEqual(mc.detect_conversion_type(OMOP_CODES), OMOP)
        self.assertEqual(mc.detect_conversion_type(MIMIC_CODES), MIMIC)

    def test_the_most_matching_events_decide_in_a_mixed_dataset(self):
        codes = OMOP_CODES + ["HOSPITAL_ADMISSION//A"]
        self.assertEqual(mc.detect_conversion_type(codes), OMOP)

    def test_codes_that_no_rule_matches_give_none(self):
        self.assertIsNone(mc.detect_conversion_type(["MEDS_BIRTH", "LOINC/1"]))
        self.assertIsNone(mc.detect_conversion_type([]))

    def test_a_tie_gives_none(self):
        codes = ["Visit/IP", "HOSPITAL_ADMISSION//A"]
        self.assertIsNone(mc.detect_conversion_type(codes))

    def test_scores_count_the_matching_events(self):
        codes = ["Visit/IP", "Visit/IP", "HOSPITAL_DISCHARGE//X", "LAB/1"]
        self.assertEqual(mc.score_conversion_types(codes), {MIMIC: 1, OMOP: 2})


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class ResolveConversionTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.omop_db = build_database(cls.root, "omop", OMOP_CODES)
        cls.mimic_db = build_database(cls.root, "mimic", MIMIC_CODES)
        cls.stay_db = build_database(cls.root, "stay", STAY_CODES)
        cls.other_db = build_database(
            cls.root, "other", ["MEDS_BIRTH", "LOINC/1", "RxNorm/2"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_codes_are_read_from_a_real_database(self):
        codes = mc.sample_event_codes(self.mimic_db)
        self.assertEqual(set(codes), set(MIMIC_CODES))

    def test_nothing_has_to_be_passed_for_a_mimic_dataset(self):
        args = data_args(self.mimic_db)
        # the default of the arguments
        self.assertEqual(args.meds_to_cehrbert_conversion_type, OMOP)
        self.assertEqual(mc.resolve_meds_conversion_type(args), MIMIC)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, MIMIC)

    def test_an_omop_dataset_keeps_the_omop_conversion(self):
        args = data_args(self.omop_db)
        self.assertEqual(mc.resolve_meds_conversion_type(args), OMOP)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, OMOP)

    def test_a_conversion_that_is_asked_for_wins_over_the_data(self):
        args = data_args(self.omop_db, MIMIC)
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.assertEqual(mc.resolve_meds_conversion_type(args), MIMIC)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, MIMIC)
        self.assertIn("looks like", logs.output[0])

    def test_data_that_no_conversion_matches_keeps_the_default_and_says_so(self):
        args = data_args(self.other_db)
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.assertEqual(mc.resolve_meds_conversion_type(args), OMOP)
        self.assertIn("No MEDS conversion matches", logs.output[0])

    def test_the_resolved_conversion_keeps_a_mimic_stay_as_one_inpatient_visit(self):
        def visits(conversion_type):
            conversion = get_meds_to_cehrbert_conversion_cls(
                conversion_type, default_visit_id=1
            )
            with meds_reader.SubjectDatabase(self.stay_db) as database:
                return convert_one_patient(database[1], conversion)["visits"]

        # the default conversion cuts the stay at its admission and its discharge
        self.assertEqual(
            [v["visit_type"] for v in visits(OMOP)], ["Visit/OP", "Visit/OP"]
        )
        resolved = mc.resolve_meds_conversion_type(data_args(self.stay_db))
        stay = visits(resolved)
        self.assertEqual([v["visit_type"] for v in stay], ["Visit/IP"])
        self.assertEqual(
            (stay[0]["visit_start_datetime"].day, stay[0]["visit_end_datetime"].day),
            (2, 3),
        )
