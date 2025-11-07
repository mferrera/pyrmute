"""Tests for _model_utils.py."""

import pytest
from pydantic import BaseModel, RootModel

from pyrmute._model_utils import get_root_annotation, is_root_model


def test_is_root_model() -> None:
    """Tests that is_root_model correctly detects."""

    class A(BaseModel):
        val: int

    class B(RootModel[list[int]]):
        root: list[int]

    class C:
        pass

    assert is_root_model(A) is False
    assert is_root_model(B) is True
    assert is_root_model(C) is False  # type: ignore[arg-type]


def test_get_root_annotation_not_rootmodel() -> None:
    """Tests getting a RootModel annotation from a not-RootModel."""

    class A(BaseModel):
        val: int

    with pytest.raises(ValueError, match="not a RootModel"):
        get_root_annotation(A)


def test_get_root_annotation_rootmodel() -> None:
    """Tests getting a RootModel annotation from a not-RootModel."""

    class B(RootModel[list[int]]):
        root: list[int]

    assert get_root_annotation(B) == list[int]
