"""Utilities for working with BaseModel and RootModel."""

from typing import Any

from pydantic import BaseModel, RootModel


def is_root_model(model_class: type[BaseModel]) -> bool:
    """Check if a model class is a RootModel.

    Args:
        model_class: The model class to check.

    Returns:
        True if the model is a RootModel, False otherwise.
    """
    return issubclass(model_class, RootModel)


def get_root_annotation(model_class: type[BaseModel]) -> Any:
    """Get the root type annotation from a RootModel.

    Args:
        model_class: A RootModel class.

    Returns:
        The type annotation of the root field.

    Raises:
        ValueError: If the model is not a RootModel.
    """
    if not is_root_model(model_class):
        raise ValueError(f"{model_class.__name__} is not a RootModel")

    root_field = model_class.model_fields.get("root")
    if root_field is None:
        raise ValueError(f"{model_class.__name__} has no root field")

    return root_field.annotation
