import unittest
from datetime import datetime, timezone

from meds_to_cehr.token_conventions import (
    ETHOS_TIME_BUCKET_TO_DAYS,
    construct_age_sequence,
    construct_time_sequence,
    encode_demographics,
    extract_time_interval_in_days,
    is_att_token,
    is_ethos_time_bucket_token,
    is_inpatient_att_token,
    is_visit_start,
    multiple_of_10,
)


class TestTokenFunctions(unittest.TestCase):
    def test_is_visit_start(self):
        self.assertTrue(is_visit_start("VS"))
        self.assertFalse(is_visit_start("D1"))

    def test_is_att_token(self):
        self.assertTrue(is_att_token("D7"))
        self.assertTrue(is_att_token("VS-D7"))
        self.assertFalse(is_att_token("random"))

    def test_is_inpatient_att_token(self):
        self.assertTrue(is_inpatient_att_token("VS-D7"))
        self.assertTrue(is_inpatient_att_token("VS-D7-VE"))
        self.assertTrue(is_inpatient_att_token("i-D7"))
        self.assertTrue(is_inpatient_att_token("i-D7-VE"))
        self.assertFalse(is_inpatient_att_token("V-D7"))
        self.assertFalse(is_inpatient_att_token("D7-VE"))


class TestTimeIntervalFunctions(unittest.TestCase):
    def test_extract_time_interval_in_days(self):
        self.assertEqual(extract_time_interval_in_days("D7"), 7)
        self.assertEqual(extract_time_interval_in_days("W1"), 7)
        self.assertEqual(extract_time_interval_in_days("M1"), 30)
        self.assertEqual(extract_time_interval_in_days("Y1"), 365)

    def test_extract_time_interval_invalid(self):
        with self.assertRaises(ValueError):
            extract_time_interval_in_days("invalid")


class TestEthosTimeTokens(unittest.TestCase):
    def test_every_bucket_and_its_inpatient_variant_converts_to_days(self):
        for bucket, days in ETHOS_TIME_BUCKET_TO_DAYS.items():
            for token in (bucket, f"i-{bucket}"):
                self.assertTrue(is_ethos_time_bucket_token(token), token)
                self.assertTrue(is_att_token(token), token)
                self.assertEqual(extract_time_interval_in_days(token), days, token)

    def test_six_month_token_is_180_days_and_repeats_add_up(self):
        self.assertAlmostEqual(extract_time_interval_in_days("=6mt"), 180, delta=1)
        total = sum(extract_time_interval_in_days(t) for t in ["=6mt"] * 3)
        self.assertAlmostEqual(total, 540, delta=3)

    def test_legacy_bucket_labels_still_parse(self):
        self.assertEqual(extract_time_interval_in_days("2w-1mt"), 22)
        self.assertEqual(extract_time_interval_in_days("i-12h-1d"), 1)

    def test_standard_att_tokens_are_unchanged(self):
        self.assertEqual(extract_time_interval_in_days("D7"), 7)
        self.assertEqual(extract_time_interval_in_days("W2"), 14)
        self.assertEqual(extract_time_interval_in_days("i-D3"), 3)
        self.assertEqual(extract_time_interval_in_days("LT"), 365 * 3)
        self.assertFalse(is_ethos_time_bucket_token("D7"))

    def test_non_time_tokens_are_not_time_buckets(self):
        for token in ("ICD9CM/0/311", "Visit/OP", "VALUE_BIN/8", "Gender/F"):
            self.assertFalse(is_ethos_time_bucket_token(token), token)


class TestSequenceReconstruction(unittest.TestCase):
    def test_ages_grow_with_the_time_tokens(self):
        concept_ids = ["year:2020", "age:40", "Gender/F", "Race/White", "[VS]", "9201", "[VE]", "W2", "[VS]", "9201", "[VE]", "LT"]
        ages = construct_age_sequence(concept_ids)
        self.assertEqual(len(ages), len(concept_ids))
        self.assertEqual(ages[:7], [40] * 7)
        # 14 days later the patient is still 40, three years after that 41
        self.assertEqual(ages[8], 40)
        self.assertEqual(ages[-1], 40 + (14 + 365 * 3) // 365)

    def test_given_ages_are_returned_as_they_are(self):
        self.assertEqual(construct_age_sequence(["year:2020", "age:40"], [1, 2]), [1, 2])

    def test_a_missing_age_token_gives_zeros(self):
        self.assertEqual(construct_age_sequence(["year:2020", "Gender/F", "x"]), [0, 0, 0])

    def test_an_age_that_is_not_a_number_is_zero(self):
        self.assertEqual(construct_age_sequence(["year:2020", "age:abc", "D1"]), [0, 0, 0])

    def test_times_start_in_the_year_token_and_advance_with_the_time_tokens(self):
        times = construct_time_sequence(["year:2020", "age:40", "D3", "x", "W1", "y"])
        start = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        day = 24 * 3600
        self.assertEqual(times, [start, start, start + 3 * day, start + 3 * day, start + 10 * day, start + 10 * day])

    def test_without_a_year_token_the_patient_is_anchored_in_1985(self):
        self.assertEqual(
            construct_time_sequence(["Gender/F"]), [datetime(1985, 1, 1, tzinfo=timezone.utc).timestamp()]
        )

    def test_given_times_are_returned_as_they_are(self):
        self.assertEqual(construct_time_sequence(["year:2020"], [5.0]), [5.0])


class TestSmallHelpers(unittest.TestCase):
    def test_multiple_of_10_is_the_next_multiple_above(self):
        self.assertEqual([multiple_of_10(n) for n in (0, 9, 10, 25)], [10, 10, 20, 30])

    def test_demographics_are_encoded_into_one_number_that_stays_unique(self):
        codes = {encode_demographics(a, g, r) for a in (0, 1, 199) for g in (0, 1, 9) for r in (0, 1, 9)}
        self.assertEqual(len(codes), 27)

    def test_demographics_out_of_range_are_refused(self):
        with self.assertRaises(AssertionError):
            encode_demographics(200, 0, 0)


if __name__ == "__main__":
    unittest.main()
