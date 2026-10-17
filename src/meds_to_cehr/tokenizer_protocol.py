"""What the tokenization mappings need of a tokenizer, so that this package does not depend on any model repo."""

from typing import Any, Dict, Iterable, List, Protocol, Sequence, Union, runtime_checkable


@runtime_checkable
class SequenceTokenizer(Protocol):
    """
    The tokenizer of a patient sequence (CehrGptTokenizer is one).

    The mappings that tokenize a sequence (HFCehrGptTokenizationMapping and HFFineTuningMapping) call these members
    and nothing else of the tokenizer.
    """

    @property
    def lab_token_ids(self) -> Iterable[int]:
        """The ids of the tokens of lab values."""

    @property
    def numeric_concept_ids(self) -> Iterable[str]:
        """The concepts whose values are numeric and are binned."""

    def get_vocab(self) -> Dict[str, int]:
        """The tokens of the vocabulary and their ids."""

    def encode(self, concept_ids: Sequence[str], **kwargs: Any) -> Sequence[int]:
        """The ids of a sequence of tokens."""

    def decode(self, concept_token_ids: List[int], skip_special_tokens: bool = True, **kwargs: Any) -> List[str]:
        """The tokens of a sequence of ids."""

    def normalize(self, concept_id: str, unit: str, concept_value: Union[float, str]) -> str:
        """The bin of a numeric value of a concept."""

    def encode_value(self, concept_values: Sequence[str]) -> Sequence[int]:
        """The ids of a sequence of value bins."""
