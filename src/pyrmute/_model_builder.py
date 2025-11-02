"""Utilities for dynamic model composition with nested version resolution."""

from collections.abc import Callable
from typing import Any, cast, get_args, get_origin

from pydantic import BaseModel, create_model

from ._type_inspector import TypeInspector
from .model_version import ModelVersion


def extract_base_type(annotation: Any) -> Any:  # noqa: PLR0911
    """Extract base type from Optional, List, Union, etc.

    Args:
        annotation: Type annotation to extract from

    Returns:
        Base type if found, None otherwise
    """
    if TypeInspector.is_optional_type(annotation):
        non_none_args = TypeInspector.get_non_none_union_args(annotation)
        if len(non_none_args) == 1:
            return extract_base_type(non_none_args[0])
        return None

    origin = get_origin(annotation)

    if TypeInspector.is_list_like(origin):
        args = get_args(annotation)
        if args:
            return extract_base_type(args[0])

    if TypeInspector.is_variable_length_tuple(annotation):
        element_types = TypeInspector.get_tuple_element_types(annotation)
        if element_types:
            return extract_base_type(element_types[0])

    if TypeInspector.is_dict_like(origin, annotation):
        args = get_args(annotation)
        if len(args) == 2:  # noqa: PLR2004
            return extract_base_type(args[1])
        return None

    if isinstance(annotation, type):
        return annotation

    return None


def _resolve_union_type(
    args: tuple[type, ...], nested_models: dict[str, type[BaseModel]], field_path: str
) -> Any:
    """Resolve union type arguments, replacing nested models."""
    resolved_args = []
    changed = False

    for arg in args:
        if arg is type(None):
            resolved_args.append(arg)
        elif TypeInspector.is_base_model(arg):
            if field_path in nested_models:
                resolved_args.append(nested_models[field_path])
                changed = True
            else:
                resolved_args.append(arg)
        else:
            resolved_args.append(arg)

    if not changed:
        return None

    if len(resolved_args) == 2 and type(None) in resolved_args:  # noqa: PLR2004
        non_none = next(a for a in resolved_args if a is not type(None))
        return non_none | None

    result = resolved_args[0]
    for arg in resolved_args[1:]:
        result = result | arg  # type: ignore[assignment]
    return result


def _resolve_list_type(
    origin: type,
    args: tuple[type, ...],
    nested_models: dict[str, type[BaseModel]],
    field_path: str,
) -> Any:
    """Resolve list-like type, replacing nested models."""
    if not args:
        return None

    if TypeInspector.is_base_model(args[0]) and field_path in nested_models:
        return origin[nested_models[field_path]]  # type: ignore[index]

    resolved_inner = resolve_nested_model_type(args[0], nested_models, field_path)
    if resolved_inner is not args[0]:
        return origin[resolved_inner]  # type: ignore[index]

    return None


def _resolve_tuple_type(
    args: tuple[type, ...], nested_models: dict[str, type[BaseModel]], field_path: str
) -> Any:
    """Resolve variable-length tuple type."""
    element_types = TypeInspector.get_tuple_element_types(
        tuple[args]  # type: ignore[valid-type]
    )
    if not element_types:
        return None

    if TypeInspector.is_base_model(element_types[0]) and field_path in nested_models:
        return tuple[nested_models[field_path], ...]  # type: ignore[valid-type]

    resolved_inner = resolve_nested_model_type(
        element_types[0], nested_models, field_path
    )
    if resolved_inner is not element_types[0]:
        return tuple[resolved_inner, ...]  # type: ignore[valid-type]

    return None


def _resolve_fixed_tuple_type(
    args: tuple[Any, ...], nested_models: dict[str, type[BaseModel]], field_path: str
) -> Any:
    """Resolve fixed-length tuple type."""
    if not args or args[-1] is Ellipsis:
        return None

    resolved_args = []
    changed = False

    for arg in args:
        if TypeInspector.is_base_model(arg) and field_path in nested_models:
            resolved_args.append(nested_models[field_path])
            changed = True
        else:
            resolved_args.append(arg)

    if changed:
        return tuple[tuple(resolved_args)]  # type: ignore[misc]

    return None


def _resolve_dict_type(
    args: tuple[type, ...], nested_models: dict[str, type[BaseModel]], field_path: str
) -> Any:
    """Resolve dict type, replacing nested value models."""
    if len(args) != 2:  # noqa: PLR2004
        return None

    key_type, value_type = args

    if TypeInspector.is_base_model(value_type) and field_path in nested_models:
        return dict[key_type, nested_models[field_path]]  # type: ignore[valid-type]

    resolved_value = resolve_nested_model_type(value_type, nested_models, field_path)
    if resolved_value is not value_type:
        return dict[key_type, resolved_value]  # type: ignore[valid-type]

    return None


def resolve_nested_model_type(  # noqa: PLR0911, C901
    field_type: type, nested_models: dict[str, type[BaseModel]], field_path: str
) -> Any:
    """Resolve a field type, replacing nested models with pinned versions.

    Args:
        field_type: Original field type annotation
        nested_models: Map of field paths to resolved model classes
        field_path: Current field path for lookup

    Returns:
        Resolved type with pinned nested models

    Note:
        Returns contain generic type expressions (unions, subscripted generics) which
        mypy represents as 'Any'. We use type: ignore comments and cast() where
        appropriate since these are valid runtime type objects.
    """
    if field_path in nested_models and TypeInspector.is_base_model(field_type):
        return nested_models[field_path]

    origin = get_origin(field_type)
    args = get_args(field_type)

    if TypeInspector.is_union_type(origin):
        resolved = _resolve_union_type(args, nested_models, field_path)
        if resolved:
            return resolved

    if TypeInspector.is_list_like(origin):
        resolved = _resolve_list_type(
            origin,  # type: ignore[arg-type]
            args,
            nested_models,
            field_path,
        )
        if resolved:
            return resolved

    if TypeInspector.is_variable_length_tuple(field_type):
        resolved = _resolve_tuple_type(args, nested_models, field_path)
        if resolved:
            return resolved

    if origin is tuple:
        resolved = _resolve_fixed_tuple_type(args, nested_models, field_path)
        if resolved:
            return resolved

    if TypeInspector.is_dict_like(origin, field_type):
        resolved = _resolve_dict_type(args, nested_models, field_path)
        if resolved:
            return resolved

    return field_type


def build_composite_model(
    name: str,
    structure: type[BaseModel],
    nested_models: dict[str, type[BaseModel]],
    module: str | None = None,
) -> type[BaseModel]:
    """Build a model with specific nested model versions.

    Args:
        name: Name for the new model class
        structure: Base model structure with generic nested types
        nested_models: Map of field paths to specific model versions
        module: Module name for the generated class (helps with pickling/IDE)

    Returns:
        New model class with resolved nested types and proper annotations

    Example:
    ```python
        class UserStructure(BaseModel):
            name: str
            address: Address

        composed = build_composite_model(
            "User_v2_0_0",
            UserStructure,
            {"address": AddressV2},
            module="myapp.models"
        )
        # Result: User_v2_0_0 with address: AddressV2
    ```
    """
    # Actually a dict[str, tuple[str, FieldInfo]]
    fields: dict[str, Any] = {}

    for field_name, field_info in structure.model_fields.items():
        field_path = field_name

        if field_info.annotation is None:
            fields[field_name] = (Any, field_info)
            continue

        resolved_type = resolve_nested_model_type(
            field_info.annotation, nested_models, field_path
        )
        fields[field_name] = (resolved_type, field_info)

    model_module = module or structure.__module__
    composed_model = create_model(
        name,
        __base__=structure,  # Inherit from structure to get validators
        __module__=model_module,
        **fields,
    )

    if structure.__doc__:
        composed_model.__doc__ = structure.__doc__

    if hasattr(structure, "model_config"):
        composed_model.model_config = structure.model_config.copy()

    return composed_model


def auto_detect_nested_pins(
    structure: type[BaseModel],
    get_model_info: Callable[[type[BaseModel]], tuple[str, ModelVersion] | None],
    prefix: str = "",
) -> dict[str, str]:
    """Auto-detect nested model pins recursively (transitive closure).

    This walks the entire model tree and builds a complete pin map, so parent
    models automatically detect all nested dependencies at any depth.

    Args:
        structure: Model structure to analyze
        get_model_info: Function to get (name, version) for a model class
        prefix: Current field path prefix (for recursion)

    Returns:
        Dictionary of "field.path" -> "Model:version"
    """
    pins = {}

    for field_name, field_info in structure.model_fields.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name

        field_type = extract_base_type(field_info.annotation)

        if field_type and TypeInspector.is_base_model(field_type):
            model_info = get_model_info(field_type)
            if model_info:
                model_name, model_version = model_info
                pins[field_path] = f"{model_name}:{model_version}"

                nested_pins = auto_detect_nested_pins(
                    field_type, get_model_info, prefix=field_path
                )
                pins.update(nested_pins)

    return pins


def normalize_pin_specs(
    structure: type[BaseModel],
    pins: dict[str, str],
    get_model_info: Callable[[type[BaseModel]], tuple[str, ModelVersion] | None],
) -> dict[str, str]:
    """Normalize pin specifications to "Model:version" format.

    Handles shorthand like "2.0.0" by inferring model name from field type.

    Args:
        structure: Model structure
        pins: Pin specifications (may be shorthand)
        get_model_info: Function to get (name, version) for a model class

    Returns:
        Normalized pins in "Model:version" format
    """
    normalized = {}

    for field_path, pin_spec in pins.items():
        if ":" in pin_spec:
            normalized[field_path] = pin_spec
        else:
            path_parts = field_path.split(".")
            current_type = structure

            for part in path_parts:
                if (
                    hasattr(current_type, "model_fields")
                    and part in current_type.model_fields
                ):
                    field_info = current_type.model_fields[part]
                    field_type = extract_base_type(field_info.annotation)
                    if field_type and TypeInspector.is_base_model(field_type):
                        current_type = field_type
                    else:
                        break

            if TypeInspector.is_base_model(current_type):
                model_info = get_model_info(current_type)
                if model_info:
                    model_name, _ = model_info
                    normalized[field_path] = f"{model_name}:{pin_spec}"
                    continue

            field_name = field_path.split(".")[-1]
            normalized[field_path] = f"{field_name.title()}:{pin_spec}"

    return normalized


def merge_nested_pins(
    inherited_pins: dict[str, tuple[str, ModelVersion]],
    detected_pins: dict[str, str],
    explicit_pins: dict[str, str],
) -> dict[str, tuple[str, ModelVersion]]:
    """Merge nested pins from multiple sources with proper precedence.

    Precedence order (lowest to highest):
    1. Inherited pins (from inherit_from)
    2. Auto-detected pins (from type hints)
    3. Explicit pins (from kwargs)

    Args:
        inherited_pins: Pins from parent version
        detected_pins: Auto-detected pins (format: "Model:version")
        explicit_pins: Explicitly specified pins (format: "Model:version")

    Returns:
        Merged pins as {field_path: (model_name, model_version)}
    """
    merged: dict[str, tuple[str, ModelVersion]] = {}
    merged.update(inherited_pins)

    for field_path, pin_spec in detected_pins.items():
        if field_path not in explicit_pins:  # Don't override explicit
            model_name, version_str = pin_spec.split(":", 1)
            merged[field_path] = (model_name, ModelVersion.parse(version_str))

    for field_path, pin_spec in explicit_pins.items():
        model_name, version_str = pin_spec.split(":", 1)
        merged[field_path] = (model_name, ModelVersion.parse(version_str))

    return merged


def format_type_annotation_for_stub(  # noqa: PLR0911, PLR0912, C901
    annotation: Any,
    registry_lookup: Callable[[type[BaseModel]], tuple[str, ModelVersion] | None],
) -> str:
    """Format a type annotation as a string for .pyi stub files.

    Args:
        annotation: Type annotation to format
        registry_lookup: Function to get (name, version) for a model class

    Returns:
        String representation suitable for stub files
    """
    if annotation is type(None):
        return "None"

    if TypeInspector.is_base_model(annotation):
        model_info = registry_lookup(annotation)
        if model_info:
            name, ver = model_info
            return f"{name}V{ver.major}_{ver.minor}_{ver.patch}"
        return cast("str", annotation.__name__)

    origin = get_origin(annotation)
    if TypeInspector.is_union_type(origin):
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:  # noqa: PLR2004
            non_none = next(arg for arg in args if arg is not type(None))
            inner_str = format_type_annotation_for_stub(non_none, registry_lookup)
            return f"{inner_str} | None"
        arg_strs = [
            format_type_annotation_for_stub(arg, registry_lookup) for arg in args
        ]
        return " | ".join(arg_strs)

    if TypeInspector.is_list_like(origin):
        args = get_args(annotation)
        origin_name = origin.__name__ if hasattr(origin, "__name__") else "list"
        if args:
            inner_str = format_type_annotation_for_stub(args[0], registry_lookup)
            return f"{origin_name}[{inner_str}]"
        return f"{origin_name}[Any]"

    if TypeInspector.is_dict_like(origin, annotation):
        args = get_args(annotation)
        if len(args) == 2:  # noqa: PLR2004
            key_str = format_type_annotation_for_stub(args[0], registry_lookup)
            val_str = format_type_annotation_for_stub(args[1], registry_lookup)
            return f"dict[{key_str}, {val_str}]"
        return "dict[Any, Any]"

    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return "tuple"
        if len(args) == 2 and args[1] is Ellipsis:  # noqa: PLR2004
            inner_str = format_type_annotation_for_stub(args[0], registry_lookup)
            return f"tuple[{inner_str}, ...]"
        arg_strs = [
            format_type_annotation_for_stub(arg, registry_lookup) for arg in args
        ]
        return f"tuple[{', '.join(arg_strs)}]"

    if origin is not None:
        args = get_args(annotation)
        if args:
            origin_name = getattr(origin, "__name__", str(origin))
            arg_strs = [
                format_type_annotation_for_stub(arg, registry_lookup) for arg in args
            ]
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return getattr(origin, "__name__", str(origin))

    if hasattr(annotation, "__name__"):
        return cast("str", annotation.__name__)

    return "Any"


def generate_model_stub(
    model_class: type[BaseModel],
    model_name: str,
    version: ModelVersion,
    registry_lookup: Callable[[type[BaseModel]], tuple[str, ModelVersion] | None],
    include_docstring: bool = True,
) -> str:
    """Generate a .pyi stub for a single model version.

    Args:
        model_class: The Pydantic model class
        model_name: Name of the model
        version: Model version
        registry_lookup: Function to get (name, version) for nested models
        include_docstring: Whether to include the model's docstring

    Returns:
        Python stub code as a string

    Example:
    ```python
        stub = generate_model_stub(UserV1, "User", ModelVersion(1, 0, 0), ...)
        # Returns:
        # class UserV1_0_0(BaseModel):
        #     name: str
        #     address: AddressV1_0_0
    ```
    """
    lines = []

    class_name = f"{model_name}V{version.major}_{version.minor}_{version.patch}"
    lines.append(f"class {class_name}(BaseModel):")

    if include_docstring and model_class.__doc__:
        doc = model_class.__doc__.strip()
        if "\n" in doc:
            lines.append('    """')
            lines.extend(f"    {line}" for line in doc.split("\n"))
            lines.append('    """')
        else:
            lines.append(f'    """{doc}"""')

    has_fields = False
    for field_name, field_info in model_class.model_fields.items():
        has_fields = True
        type_str = format_type_annotation_for_stub(
            field_info.annotation, registry_lookup
        )

        if field_info.default is not None or field_info.default_factory is not None:
            lines.append(f"    {field_name}: {type_str} = ...")
        else:
            lines.append(f"    {field_name}: {type_str}")

    if not has_fields:
        lines.append("    pass")

    return "\n".join(lines)


def generate_stub_file_content(
    models: list[tuple[str, ModelVersion, type[BaseModel]]],
    registry_lookup: Callable[[type[BaseModel]], tuple[str, ModelVersion] | None],
    package_name: str,
) -> str:
    """Generate complete .pyi stub file content for multiple models.

    Args:
        models: List of (model_name, version, model_class) tuples
        registry_lookup: Function to get (name, version) for nested models
        package_name: Name of the package (for header comment)

    Returns:
        Complete stub file content as a string

    Example:
    ```python
        models = [
            ("User", ModelVersion(1, 0, 0), UserV1),
            ("User", ModelVersion(2, 0, 0), UserV2),
        ]
        content = generate_stub_file_content(models, registry.get_model_info, "myapp")
    ```
    """
    lines = []

    lines.append(f'"""Type stubs for {package_name}."""')
    lines.append("")
    lines.append("# This file is auto-generated. Do not edit manually.")
    lines.append("")

    lines.append("from pydantic import BaseModel")
    lines.append("from typing import Any")
    lines.append("")

    for i, (model_name, version, model_class) in enumerate(models):
        if i > 0:
            lines.append("")
            lines.append("")

        stub = generate_model_stub(model_class, model_name, version, registry_lookup)
        lines.append(stub)

    if models:
        lines.append("")
        lines.append("")
        lines.append("# Convenience aliases")

        by_name: dict[str, list[tuple[ModelVersion, type[BaseModel]]]] = {}
        for model_name, version, model_class in models:
            if model_name not in by_name:
                by_name[model_name] = []
            by_name[model_name].append((version, model_class))

        for model_name, versions in by_name.items():
            versions.sort(key=lambda x: x[0], reverse=True)
            latest_ver, _ = versions[0]

            full_name = (
                f"{model_name}V{latest_ver.major}_{latest_ver.minor}_{latest_ver.patch}"
            )
            lines.append(f"{model_name} = {full_name}  # Latest version")

    return "\n".join(lines)


def collect_nested_model_dependencies(
    model_class: type[BaseModel],
    registry_lookup: Callable[[type[BaseModel]], tuple[str, ModelVersion] | None],
    seen: set[str] | None = None,
) -> set[tuple[str, ModelVersion]]:
    """Collect all nested model dependencies for a model.

    This is used to determine which models need to be included in stub files.

    Args:
        model_class: Model to analyze
        registry_lookup: Function to get (name, version) for nested models
        seen: Set of model names already processed (for recursion tracking)

    Returns:
        Set of (model_name, version) tuples for all nested dependencies

    Example:
        ```python
        deps = collect_nested_model_dependencies(UserV1, registry.get_model_info)
        # Returns: {("Address", ModelVersion(1, 0, 0)), ("City", ModelVersion(1, 0, 0))}
        ```
    """
    if seen is None:
        seen = set()

    model_info = registry_lookup(model_class)
    if not model_info:
        return set()

    model_name, _ = model_info
    if model_name in seen:
        return set()

    seen.add(model_name)
    dependencies = set()

    for field_info in model_class.model_fields.values():
        field_type = extract_base_type(field_info.annotation)

        if field_type and TypeInspector.is_base_model(field_type):
            nested_info = registry_lookup(field_type)
            if nested_info:
                nested_name, nested_version = nested_info
                dependencies.add((nested_name, nested_version))

                nested_deps = collect_nested_model_dependencies(
                    field_type, registry_lookup, seen
                )
                dependencies.update(nested_deps)

    return dependencies
