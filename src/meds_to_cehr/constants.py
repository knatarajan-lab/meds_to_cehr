# Adapted from cehrbert.data_generators.hf_data_generator and cehrbert_data.const (MIT License, same copyright holder as cehrgpt).
UNKNOWN_VALUE = "Unknown"
DEFAULT_ED_CONCEPT_ID = "Visit/ER"
DEFAULT_OUTPATIENT_CONCEPT_ID = "Visit/OP"
DEFAULT_INPATIENT_CONCEPT_ID = "Visit/IP"
DEFAULT_ED_TO_INPATIENT_CONCEPT_ID = "Visit/ERIP"

# Artificial tokens
VS_TOKEN = "[VS]"
VE_TOKEN = "[VE]"
DEATH_TOKEN = "[DEATH]"
VISIT_UNKNOWN_TOKEN = "Visit/0"
DISCHARGE_UNKNOWN_TOKEN = "Discharge/0"
GENDER_UNKNOWN_TOKEN = "Gender/0"
RACE_UNKNOWN_TOKEN = "Race/0"

# Value used where a lab has no unit
NA = "N/A"

# Value bins of the tokenization of numeric values
UNKNOWN_BIN = "BIN:unknown"
NONE_BIN = "BIN:NONE"
