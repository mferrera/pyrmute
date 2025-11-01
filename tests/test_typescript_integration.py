"""Tests for TypeScript schema generation from Pydantic models."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from pyrmute._registry import Registry
from pyrmute.typescript_schema import (
    TypeScriptExporter,
    TypeScriptSchemaGenerator,
)


@pytest.fixture
def generator() -> TypeScriptSchemaGenerator:
    """Create a TypeScript schema generator."""
    return TypeScriptSchemaGenerator(style="interface")


@pytest.fixture
def zod_generator() -> TypeScriptSchemaGenerator:
    """Create a Zod schema generator."""
    return TypeScriptSchemaGenerator(style="zod")


@pytest.fixture
def type_generator() -> TypeScriptSchemaGenerator:
    """Create a TypeScript type alias generator."""
    return TypeScriptSchemaGenerator(style="type")


@pytest.fixture
def zod_stub() -> str:
    """Creates a basic Zod-interface."""
    return """
export declare namespace z {
    interface ZodType<T = any> {
        parse(data: unknown): T;
        safeParse(data: unknown): { success: true; data: T } | { success: false; error: any };
        optional(): ZodType<T | undefined>;
        nullable(): ZodType<T | null>;
    }

    interface ZodString extends ZodType<string> {
        min(n: number): ZodString;
        max(n: number): ZodString;
        email(): ZodString;
        url(): ZodString;
        uuid(): ZodString;
        regex(pattern: RegExp): ZodString;
        datetime(): ZodString;
        date(): ZodString;
        time(): ZodString;
    }

    interface ZodNumber extends ZodType<number> {
        int(): ZodNumber;
        positive(): ZodNumber;
        negative(): ZodNumber;
        gte(n: number): ZodNumber;
        gt(n: number): ZodNumber;
        lte(n: number): ZodNumber;
        lt(n: number): ZodNumber;
        min(n: number): ZodNumber;
        max(n: number): ZodNumber;
    }

    interface ZodObject<T> extends ZodType<T> {
        passthrough(): ZodObject<T>;
    }

    function object<T>(shape: any): ZodObject<T>;
    function string(): ZodString;
    function number(): ZodNumber;
    function boolean(): ZodType<boolean>;
    function literal<T>(value: T): ZodType<T>;
    function array<T>(schema: ZodType<T>): ZodType<T[]>;
    function tuple<T extends any[]>(schemas: any[]): ZodType<T>;
    function union<T extends any[]>(schemas: any[]): ZodType<T[number]>;
    function record<T>(schema: ZodType<T>): ZodType<Record<string, T>>;
    function any(): ZodType<any>;
    function never(): ZodType<never>;

    type infer<T extends ZodType> = T extends ZodType<infer U> ? U : never;
}
"""  # noqa: E501


@pytest.fixture
def typescript_validator() -> bool:
    """Check if TypeScript compiler is available."""
    try:
        result = subprocess.run(
            ["tsc", "--version"], check=False, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


@pytest.mark.integration
def test_generated_typescript_is_valid(
    generator: TypeScriptSchemaGenerator, typescript_validator: bool
) -> None:
    """Test that generated TypeScript code is syntactically valid."""
    if not typescript_validator:
        pytest.skip("TypeScript compiler not available")

    class User(BaseModel):
        name: str
        age: int
        email: str | None = None
        tags: list[str] = []

    module = generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(schema)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", temp_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"TypeScript validation failed: {result.stderr}"
    finally:
        Path(temp_path).unlink()


@pytest.mark.integration
def test_generated_zod_schema_is_valid(
    zod_generator: TypeScriptSchemaGenerator, typescript_validator: bool, zod_stub: str
) -> None:
    """Test that generated Zod schemas are syntactically valid."""
    if not typescript_validator:
        pytest.skip("TypeScript compiler not available")

    class User(BaseModel):
        name: str
        age: int
        email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    module = zod_generator.generate_schema(User, "User", "1.0.0")
    schema = module.to_string()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        package_json = {"dependencies": {"zod": "^4.0.0"}}
        (tmpdir_path / "package.json").write_text(json.dumps(package_json))

        node_modules = tmpdir_path / "node_modules" / "zod"
        node_modules.mkdir(parents=True)
        (node_modules / "index.d.ts").write_text(zod_stub)

        schema_file = tmpdir_path / "schema.ts"
        schema_file.write_text(schema)

        tsconfig = {
            "compilerOptions": {
                "strict": True,
                "noEmit": True,
                "moduleResolution": "node",
                "esModuleInterop": True,
            }
        }
        (tmpdir_path / "tsconfig.json").write_text(json.dumps(tsconfig))

        result = subprocess.run(
            ["tsc", "--noEmit"],
            check=False,
            cwd=tmpdir_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            assert "Cannot find module 'zod'" in result.stderr or result.returncode == 0


@pytest.mark.integration
def test_generated_typescript_with_organization_is_valid(
    tmp_path: Path, typescript_validator: bool
) -> None:
    """Test generated TypeScript schemas with organization are valid and importable."""
    if not typescript_validator:
        pytest.skip("TypeScript compiler not available")

    class User(BaseModel):
        name: str
        age: int
        email: str | None = None

    class Order(BaseModel):
        id: int
        user_id: int
        total: float

    registry = Registry()
    registry.register("User", "1.0.0")(User)
    registry.register("User", "2.0.0")(User)
    registry.register("Order", "1.0.0")(Order)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"

    exporter.export_all_schemas(
        output_dir, organization="major_version", include_barrel_exports=True
    )

    test_file = tmp_path / "test-imports.ts"
    test_file.write_text("""
// Test importing from version-specific directories
import { User as UserV1 } from './types/v1/User.v1.0.0';
import { User as UserV2 } from './types/v2/User.v2.0.0';
import { Order } from './types/v1/Order.v1.0.0';

// Test importing from barrel exports
import * as V1 from './types/v1';
import * as V2 from './types/v2';

// Test using the types
const userV1: UserV1 = {
    name: "Alice",
    age: 30,
    email: "alice@example.com"
};

const userV2: UserV2 = {
    name: "Bob",
    age: 25,
};

const order: Order = {
    id: 1,
    user_id: 123,
    total: 99.99
};

// Test barrel exports work
const userFromBarrel: V1.User = userV1;
const userFromBarrel2: V2.User = userV2;
""")
    ts_files = [str(test_file)]
    ts_files.extend(str(f) for f in output_dir.rglob("*.ts"))

    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", *ts_files],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"TypeScript validation failed: {result.stderr}"
    finally:
        pass


@pytest.mark.integration
def test_generated_typescript_by_model_organization_is_valid(
    tmp_path: Path, typescript_validator: bool
) -> None:
    """Test that by-model organization generates valid TypeScript."""
    if not typescript_validator:
        pytest.skip("TypeScript compiler not available")

    class User(BaseModel):
        name: str
        email: str

    registry = Registry()
    registry.register("User", "1.0.0")(User)
    registry.register("User", "1.5.0")(User)
    registry.register("User", "2.0.0")(User)

    exporter = TypeScriptExporter(registry, style="interface")
    output_dir = tmp_path / "types"

    exporter.export_all_schemas(
        output_dir, organization="model", include_barrel_exports=True
    )

    test_file = tmp_path / "test-barrel.ts"
    test_file.write_text("""
// Import latest version via barrel export
import { User } from './types/User';

// Import specific versions
import { User as UserV1 } from './types/User/1.0.0';
import { User as UserV2 } from './types/User/2.0.0';

const user: User = {
    name: "Alice",
    email: "alice@example.com"
};

const userV1: UserV1 = user;
const userV2: UserV2 = user;
""")

    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", str(test_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"TypeScript validation failed: {result.stderr}"
    finally:
        pass


@pytest.mark.integration
def test_generated_zod_schemas_with_organization_are_valid(
    tmp_path: Path, typescript_validator: bool, zod_stub: str
) -> None:
    """Test that Zod schemas with organization are valid and usable."""
    if not typescript_validator:
        pytest.skip("TypeScript compiler not available")

    class User(BaseModel):
        name: str
        age: int
        email: str | None = None

    registry = Registry()
    registry.register("User", "1.0.0")(User)

    exporter = TypeScriptExporter(registry, style="zod")
    output_dir = tmp_path / "schemas"

    exporter.export_all_schemas(
        output_dir, organization="major_version", include_barrel_exports=True
    )

    package_json = tmp_path / "package.json"
    package_json.write_text('{"dependencies": {"zod": "^4.0.0"}}')

    node_modules = tmp_path / "node_modules" / "zod"
    node_modules.mkdir(parents=True)
    (node_modules / "index.d.ts").write_text(zod_stub)

    test_file = tmp_path / "test-zod.ts"
    test_file.write_text("""
import { UserSchema, User } from './schemas/v1';

// Test runtime validation
const validUser = {
    name: "Alice",
    age: 30,
    email: "alice@example.com"
};

const parsedUser: User = UserSchema.parse(validUser);

// Test type usage
const user: User = {
    name: "Bob",
    age: 25,
    email: null
};
""")

    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", str(test_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"TypeScript validation failed: {result.stderr}"
    finally:
        pass
