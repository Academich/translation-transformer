from dataclasses import dataclass

import torch
import pytest

from src.model.mock_model import MockCopySequence
from src.translation.translators import TranslationInferenceGreedy
from src.translation.translators import TranslationInferenceBeamSearch


@dataclass(frozen=True)
class BeamSearchCase:
    max_len: int
    beam_size: int
    src: 'torch.LongTensor'
    tgt: 'torch.LongTensor'


@pytest.fixture(scope="module")
def greedy_decoder():
    return TranslationInferenceGreedy(model=MockCopySequence(),
                                      max_len=10,
                                      pad_token=MockCopySequence.pad_token,
                                      bos_token=MockCopySequence.bos_token,
                                      eos_token=MockCopySequence.eos_token)


@pytest.fixture(scope="module")
def beam_search_decoder():
    return TranslationInferenceBeamSearch(model=MockCopySequence(),
                                          max_len=0,
                                          beam_size=0,
                                          pad_token=MockCopySequence.pad_token,
                                          bos_token=MockCopySequence.bos_token,
                                          eos_token=MockCopySequence.eos_token)


GREEDY_CASES = [
    (
        torch.tensor([[1, 2, 3, 4, 10]]),
        torch.tensor([[1, 2, 3, 4, 10]])
    ),

    (
        torch.tensor([[1, 2, 3, 4, 5, 4, 3, 2, 10]]),
        torch.tensor([[1, 2, 3, 4, 5, 4, 3, 2, 10]])
    ),

    (
        torch.tensor([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10]]),
        torch.tensor([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    ),
    (
        torch.tensor([[1, 2, 2, 10, 2, 2, 2, 2, 2, 2, 2, 2, 10]]),
        torch.tensor([[1, 2, 2, 10]])
    ),

    (
        torch.tensor([[1, 6, 10]]),
        torch.tensor([[1, 6, 10]])
    ),

    (
        torch.tensor([[1, 6, 10, 0],
                      [1, 3, 9, 10],
                      [1, 4, 10, 0],
                      [1, 8, 10, 0]]),
        torch.tensor([[1, 6, 10, 0],
                      [1, 3, 9, 10],
                      [1, 4, 10, 0],
                      [1, 8, 10, 0]])
    ),

    (
        torch.tensor([[1, 2, 3, 4, 10, 0, 0],
                      [1, 2, 3, 4, 5, 10, 0],
                      [1, 2, 3, 4, 5, 6, 10]]),
        torch.tensor([[1, 2, 3, 4, 10, 0, 0],
                      [1, 2, 3, 4, 5, 10, 0],
                      [1, 2, 3, 4, 5, 6, 10]])
    )

]


@pytest.mark.parametrize('src, ground_truth', GREEDY_CASES)
def test_greedy(greedy_decoder, src, ground_truth):
    assert torch.allclose(greedy_decoder.generate(src.long()), ground_truth)


BEAM_SEARCH_CASES = [
    BeamSearchCase(
        max_len=10,
        beam_size=1,
        src=torch.tensor([[1, 2, 3, 4, 10]]).long(),
        tgt=torch.tensor([[[1, 2, 3, 4, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=1,
        src=torch.tensor([[1, 2, 3, 4, 10],
                          [1, 5, 2, 4, 10]]).long(),
        tgt=torch.tensor([[[1, 2, 3, 4, 10]],
                          [[1, 5, 2, 4, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=1,
        src=torch.tensor([[1, 6, 10]]).long(),
        tgt=torch.tensor([[[1, 6, 10]]]).long()
    ),

    BeamSearchCase(
        max_len=10,
        beam_size=2,
        src=torch.tensor([[1, 2, 3, 4, 10]]).long(),
        tgt=torch.tensor([[[1, 2, 3, 4, 10],
                           [1, 3, 3, 4, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=2,
        src=torch.tensor([[1, 7, 8, 9, 10]]).long(),
        tgt=torch.tensor([[[1, 7, 8, 9, 10],
                           [1, 8, 8, 9, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=2,
        src=torch.tensor([[1, 7, 8, 9, 10],
                          [1, 7, 8, 9, 10]]).long(),
        tgt=torch.tensor([[[1, 7, 8, 9, 10],
                           [1, 8, 8, 9, 10]],
                          [[1, 7, 8, 9, 10],
                           [1, 8, 8, 9, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=3,
        src=torch.tensor([[1, 2, 3, 4, 10]]).long(),
        tgt=torch.tensor([[[1, 2, 3, 4, 10],
                           [1, 3, 3, 4, 10],
                           [1, 2, 4, 4, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=1,
        src=torch.tensor([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10]]).long(),
        tgt=torch.tensor([[[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=2,
        src=torch.tensor([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10]]).long(),
        tgt=torch.tensor([[[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=2,
        src=torch.tensor([[1, 9, 9, 10]]).long(),
        tgt=torch.tensor([[[1, 10, 0, 0],
                           [1, 9, 9, 10]]]).long()
    ),
    BeamSearchCase(
        max_len=10,
        beam_size=2,
        src=torch.tensor([[1, 8, 9, 10]]).long(),
        tgt=torch.tensor([[[1, 8, 9, 10],
                           [1, 8, 10, 0]]]).long()
    )
]


@pytest.mark.parametrize('case', BEAM_SEARCH_CASES)
def test_beam_search(beam_search_decoder, case: 'BeamSearchCase'):
    beam_search_decoder.beam_size = case.beam_size
    beam_search_decoder.max_len = case.max_len
    assert torch.allclose(beam_search_decoder.generate(case.src),
                          case.tgt)
