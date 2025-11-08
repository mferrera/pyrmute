"""Shared utilities for inspecting Python types across schema generators."""

import types
from collections.abc import Mapping
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, RootModel
from pydantic.fields import FieldInfo


class TypeInspector:
    """Utilities for inspecting and analyzing Python type annotations.

    This class provides shared methods for type checking logic used across Avro,
    Protocol Buffer, and TypeScript schema generators.
    """

    @staticmethod
    def is_union_type(origin: Any) -> bool:
        """Check if origin represents a Union type.

        Handles both `typing.Union` and Python 3.10+ `types.UnionType`.

        Args:
            origin: Type origin from get_origin().

        Returns:
            True if this is a Union type.
        """
        if origin is Union:
            return True

        if hasattr(types, "UnionType"):
            try:
                return origin is types.UnionType
            except (ImportError, AttributeError):
                pass

        return False

    @staticmethod
    def is_optional_type(annotation: Any) -> bool:
        """Check if annotation represents an Optional type.

        An Optional type is a Union that includes None.

        Args:
            annotation: Type annotation.

        Returns:
            True if this is Optional (Union with None).
        """
        origin = get_origin(annotation)
        if TypeInspector.is_union_type(origin):
            args = get_args(annotation)
            return type(None) in args
        return False

    @staticmethod
    def get_non_none_union_args(annotation: Any) -> list[Any]:
        """Extract non-None types from a Union.

        Args:
            annotation: Type annotation, should be a Union type.

        Returns:
            List of type arguments excluding None.
        """
        origin = get_origin(annotation)
        if not TypeInspector.is_union_type(origin):
            return [annotation]

        args = get_args(annotation)
        return [arg for arg in args if arg is not type(None)]

    @staticmethod
    def is_union_requiring_oneof(annotation: Any) -> bool:
        """Check if annotation is a union type that requires oneof (for Protobuf).

        A union requires oneof representation if it has multiple non-None types.
        Optional[T] (Union with single non-None type) does not require oneof.

        Args:
            annotation: Type annotation.

        Returns:
            True if this union requires oneof representation.
        """
        origin = get_origin(annotation)
        if not TypeInspector.is_union_type(origin):
            return False

        non_none_args = TypeInspector.get_non_none_union_args(annotation)
        return len(non_none_args) > 1

    @staticmethod
    def is_list_like(origin: Any) -> bool:
        """Check if origin represents a list-like type.

        Includes list, set, and frozenset.

        Args:
            origin: Type origin from get_origin().

        Returns:
            True if this is a list-like container type.
        """
        if origin in (list, set, frozenset):
            return True

        if hasattr(origin, "__origin__"):
            return origin.__origin__ in (list, set, frozenset)

        return False

    @staticmethod
    def is_variable_length_tuple(annotation: Any) -> bool:
        """Check if tuple uses ellipsis notation (variable length).

        Args:
            annotation: Type annotation.

        Returns:
            True if this is tuple[T, ...] (homogeneous, variable length).
        """
        origin = get_origin(annotation)
        if origin is not tuple:
            return False

        args = get_args(annotation)
        return len(args) == 2 and args[1] is Ellipsis  # noqa: PLR2004

    @staticmethod
    def get_tuple_element_types(annotation: Any) -> list[Any]:
        """Extract element types from a tuple annotation.

        For variable-length tuples (tuple[T, ...]), returns single element [T]. For
        fixed tuples (tuple[A, B, C]), returns all elements [A, B, C].

        Args:
            annotation: Tuple type annotation.

        Returns:
            List of element types.
        """
        origin = get_origin(annotation)
        if origin is not tuple:
            return []

        args = get_args(annotation)
        if not args:
            return []

        if len(args) == 2 and args[1] is Ellipsis:  # noqa: PLR2004
            return [args[0]]

        return list(args)

    @staticmethod
    def is_dict_like(origin: Any, annotation: Any = None) -> bool:
        """Check if origin represents a dict-like type.

        Includes dict, Mapping, and MutableMapping.

        Args:
            origin: Type origin from get_origin().
            annotation: Optional full annotation for additional checks.

        Returns:
            True if this is a dict-like mapping type.
        """
        if annotation is dict:
            return True

        if origin in (dict, Mapping):
            return True

        return origin and isinstance(origin, type) and issubclass(origin, Mapping)

    @staticmethod
    def is_enum(python_type: Any) -> bool:
        """Check if type is a Python Enum.

        Args:
            python_type: Type to check.

        Returns:
            True if this is an Enum subclass.
        """
        return isinstance(python_type, type) and issubclass(python_type, Enum)

    @staticmethod
    def is_base_model(python_type: Any) -> bool:
        """Check if type is a Pydantic BaseModel.

        Args:
            python_type: Type to check.

        Returns:
            True if this is a BaseModel subclass.
        """
        return isinstance(python_type, type) and issubclass(python_type, BaseModel)

    @staticmethod
    def collect_nested_models(
        model: type[BaseModel], seen: set[str] | None = None
    ) -> dict[str, type[BaseModel]]:
        """Recursively collect all nested BaseModel types from a model.

        Args:
            model: Pydantic model class to scan.
            seen: Set of model names already processed (for recursion tracking).

        Returns:
            Dictionary mapping model names to model classes.
        """
        if seen is None:
            seen = set()

        if model.__name__ in seen:
            return {}

        seen.add(model.__name__)
        nested_models: dict[str, type[BaseModel]] = {}

        for field_info in model.model_fields.values():
            TypeInspector._collect_from_type(field_info.annotation, nested_models, seen)

        if hasattr(model, "model_computed_fields"):
            for computed_field_info in model.model_computed_fields.values():
                if hasattr(computed_field_info, "return_type"):
                    TypeInspector._collect_from_type(
                        computed_field_info.return_type, nested_models, seen
                    )

        return nested_models

    @staticmethod
    def _collect_from_type(  # noqa: C901
        python_type: Any, nested_models: dict[str, type[BaseModel]], seen: set[str]
    ) -> None:
        """Helper to recursively collect BaseModel types from a type annotation.

        Args:
            python_type: Type annotation to scan.
            nested_models: Dictionary to populate with found models.
            seen: Set of model names already processed.
        """
        if (
            python_type is None
            or python_type is type(None)
            or isinstance(python_type, str)
        ):
            return

        if TypeInspector.is_base_model(python_type):
            if python_type.__name__ not in nested_models:
                nested_models[python_type.__name__] = python_type
                deeper_models = TypeInspector.collect_nested_models(python_type, seen)
                nested_models.update(deeper_models)
            return

        origin = get_origin(python_type)
        args = get_args(python_type)

        if TypeInspector.is_union_type(origin):
            for arg in args:
                if arg is not type(None):
                    TypeInspector._collect_from_type(arg, nested_models, seen)
            return

        if TypeInspector.is_list_like(origin):
            for arg in args:
                TypeInspector._collect_from_type(arg, nested_models, seen)
            return

        if TypeInspector.is_dict_like(origin, python_type):
            for arg in args:
                TypeInspector._collect_from_type(arg, nested_models, seen)
            return

    @staticmethod
    def get_numeric_constraints(field_info: FieldInfo) -> dict[str, int | None]:
        """Extract numeric constraints from field metadata.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            Dictionary with keys: 'ge', 'gt', 'le', 'lt' (greater/less than).
            Values are None if constraint not present.
        """
        constraints: dict[str, int | None] = {
            "ge": None,
            "gt": None,
            "le": None,
            "lt": None,
        }

        if not hasattr(field_info, "metadata") or not field_info.metadata:
            return constraints

        for constraint in field_info.metadata:
            if hasattr(constraint, "ge") and constraint.ge is not None:
                constraints["ge"] = constraint.ge
            if hasattr(constraint, "gt") and constraint.gt is not None:
                constraints["gt"] = constraint.gt
            if hasattr(constraint, "le") and constraint.le is not None:
                constraints["le"] = constraint.le
            if hasattr(constraint, "lt") and constraint.lt is not None:
                constraints["lt"] = constraint.lt

        return constraints

    @staticmethod
    def get_string_constraints(field_info: FieldInfo) -> dict[str, Any]:
        """Extract string constraints from field metadata.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            Dictionary with keys: 'pattern', 'min_length', 'max_length'.
            Values are None if constraint not present.
        """
        constraints: dict[str, Any] = {
            "pattern": None,
            "min_length": None,
            "max_length": None,
        }

        if not hasattr(field_info, "metadata") or not field_info.metadata:
            return constraints

        for constraint in field_info.metadata:
            if hasattr(constraint, "pattern") and constraint.pattern is not None:
                constraints["pattern"] = constraint.pattern
            if hasattr(constraint, "min_length") and constraint.min_length is not None:
                constraints["min_length"] = constraint.min_length
            if hasattr(constraint, "max_length") and constraint.max_length is not None:
                constraints["max_length"] = constraint.max_length

        return constraints

    @staticmethod
    def can_fit_in_32bit_int(field_info: FieldInfo) -> bool:
        """Determine if integer field constraints fit in 32-bit signed integer.

        Checks if both minimum and maximum constraints (if present) are within
        the range of a 32-bit signed integer: -2^31 to 2^31-1.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            True if constraints guarantee the value fits in 32 bits.
            False if constraints are too large or not present (err on safe side).
        """
        constraints = TypeInspector.get_numeric_constraints(field_info)

        minimum = None
        if constraints["ge"] is not None:
            minimum = constraints["ge"]
        elif constraints["gt"] is not None:
            minimum = constraints["gt"] + 1

        maximum = None
        if constraints["le"] is not None:
            maximum = constraints["le"]
        elif constraints["lt"] is not None:
            maximum = constraints["lt"] - 1

        return (
            minimum is not None
            and minimum >= -(2**31)
            and maximum is not None
            and maximum <= (2**31 - 1)
        )

    @staticmethod
    def is_unsigned_int(field_info: FieldInfo) -> bool:
        """Determine if integer field constraints guarantee unsigned value.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            True if constraints guarantee the value is non-negative (>= 0).
        """
        constraints = TypeInspector.get_numeric_constraints(field_info)
        if constraints["ge"] is not None and constraints["ge"] >= 0:
            return True
        return constraints["gt"] is not None and constraints["gt"] >= -1

    @staticmethod
    def can_fit_in_32bit_uint(field_info: FieldInfo) -> bool:
        """Determine if unsigned integer fits in 32-bit unsigned integer.

        Checks if the maximum constraint (if present) is within 0 to 2^32-1.

        Args:
            field_info: Pydantic field info with metadata.

        Returns:
            True if constraints guarantee the value fits in 32-bit unsigned.
        """
        constraints = TypeInspector.get_numeric_constraints(field_info)

        if not TypeInspector.is_unsigned_int(field_info):
            return False

        maximum = None
        if constraints["le"] is not None:
            maximum = constraints["le"]
        elif constraints["lt"] is not None:
            maximum = constraints["lt"] - 1

        return maximum is not None and maximum <= (2**32 - 1)

    @staticmethod
    def is_root_model(model_class: type[BaseModel]) -> bool:
        """Check if a model class is a RootModel.

        Args:
            model_class: The model class to check.

        Returns:
            True if the model is a RootModel, False otherwise.
        """
        return issubclass(model_class, RootModel)

    @staticmethod
    def get_root_annotation(model_class: type[BaseModel]) -> Any:
        """Get the root type annotation from a RootModel.

        Args:
            model_class: A RootModel class.

        Returns:
            The type annotation of the root field.

        Raises:
            ValueError: If the model is not a RootModel.
        """
        if not TypeInspector.is_root_model(model_class):
            raise ValueError(f"{model_class.__name__} is not a RootModel")

        root_field = model_class.model_fields.get("root")
        if root_field is None:
            raise ValueError(f"{model_class.__name__} has no root field")

        return root_field.annotation
