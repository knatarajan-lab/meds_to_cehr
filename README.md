# meds_to_cehr

Convert [MEDS](https://github.com/Medical-Event-Data-Standard/meds) datasets into the patient sequences of
[CEHR-GPT](https://github.com/knatarajan-lab/cehrgpt). The sequences of
[CEHR-BERT](https://github.com/cumc-dbmi/cehr-bert) will move here later.

The package is what is left of the data side of CEHR-GPT once the model is taken away. It does not depend on it, so a
benchmark, or another model, can use the conversion alone.

## What it does

```
MEDS dataset ──► meds_reader database ──► visits (CehrBertPatient) ──► CEHR-GPT sequence ──► token ids
   (events)     meds_reader_convert        meds_to_cehr.meds           meds_to_cehr.mapping              your tokenizer
```

* `meds_to_cehr.meds` groups the events of a patient into visits. The rules that find the visits depend on how the
  dataset codes them (`MedsToBertMimic4` for MIMIC-IV, `MedsToCehrbertOMOP` for OMOP-derived data).
  **The rules are chosen from the data**: nothing has to be passed to say what shape a dataset has
  (`meds_to_cehr.meds.conversion_detection`). A rule set can still be named explicitly.
* `meds_to_cehr.mapping` turns a patient into the sequence of CEHR-GPT: artificial time tokens between visits,
  visit start and end tokens, ages, times, and the numeric values of labs. `mapping.cehrgpt` has the mappings and
  `mapping.cehrgpt_datasets` applies them to a dataset.
* `meds_to_cehr.tokenizer_protocol.SequenceTokenizer` is what the tokenizing mappings need of a tokenizer. Any tokenizer
  with those few members works, and this package does not import one.

## Usage

```python
from meds_to_cehr.arguments import DataTrainingArguments
from meds_to_cehr.mapping.cehrgpt import MedToCehrGPTDatasetMapping
from meds_to_cehr.mapping.cehrgpt_datasets import create_cehrgpt_pretraining_dataset
from meds_to_cehr.meds.conversion_detection import resolve_meds_conversion_type
from meds_to_cehr.meds.meds_utils import create_dataset_from_meds_reader

args = DataTrainingArguments(
    data_folder="MEDS_READER_DATABASE",  # made with meds_reader_convert
    dataset_prepared_path="PREPARED",
    is_data_in_meds=True,
    include_auxiliary_token=True,
)
resolve_meds_conversion_type(args)  # looks at the event codes; leaves a conversion you set alone
patients = create_dataset_from_meds_reader(args, dataset_mappings=[MedToCehrGPTDatasetMapping(args)])
tokenized = create_cehrgpt_pretraining_dataset(patients, tokenizer, args)  # tokenizer: a SequenceTokenizer
```

`DataTrainingArguments` is a plain dataclass, so it also works with `transformers.HfArgumentParser`.

## Adding a conversion

A conversion is a subclass of `MedsToCehrBertConversion` that says which codes are an ED registration, an admission,
a discharge and a visit. Subclasses are found by the detection, which scores each one with its own rules.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests that read a MEDS dataset need the `meds_reader_convert` script that comes with `meds_reader`, and are
skipped without it.

## Origin and license

MIT License. Much of the code was moved here from [cehrbert](https://github.com/cumc-dbmi/cehr-bert),
`cehrbert_data` and [cehrgpt](https://github.com/knatarajan-lab/cehrgpt), which have the same authors and license.
The files say where they come from.
