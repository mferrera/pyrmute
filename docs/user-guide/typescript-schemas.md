# TypeScript Schema Generation

pyrmute can generate [TypeScript](https://www.typescriptlang.org/) type
definitions and [Zod](https://zod.dev/) validation schemas for all your model
versions. TypeScript types provide compile-time type safety for frontend
applications, while Zod schemas enable runtime validation. This guide covers
TypeScript schema generation, type mapping, and integration with modern web
development tooling.

!!! note
    TypeScript uses `.ts` files for type definitions and validation schemas.
    Throughout this guide, we use "schema" to refer to these TypeScript type
    definitions for consistency with the rest of pyrmute's API and
    documentation.

## Why TypeScript?

TypeScript schemas are used for:

- **Frontend type safety** - Catch type errors at compile time
- **API contracts** - Share types between backend and frontend
- **IDE support** - Autocomplete and inline documentation
- **Runtime validation** - Zod schemas validate data at runtime

**TypeScript vs JSON Schema:**

| Feature | JSON Schema | TypeScript |
|---------|-------------|------------|
| Use case | Runtime validation | Compile-time types |
| Type safety | Runtime only | Compile-time + runtime (Zod) |
| IDE support | Limited | Excellent |
| Performance | Validation overhead | Zero runtime cost (types only) |
| Ecosystem | REST APIs, OpenAPI | Frontend apps, Node.js |
| Code generation | No | Native TypeScript |

## Basic TypeScript Schema Generation

Generate a TypeScript interface for any registered model:

```python
from pydantic import BaseModel, Field
from pyrmute import ModelManager

manager = ModelManager()


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """User account information."""
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: int = Field(ge=0, le=150, description="User's age in years")


# Generate TypeScript interface
ts_schema = manager.get_typescript_schema("User", "1.0.0")
print(ts_schema)
```

Output:

```typescript
/**
 * User account information.
 */
export interface UserV1_0_0 {
  /** User's full name */
  name: string;
  /** User's email address */
  email: string;
  /** User's age in years */
  age: number;
}
```

### Schema Styles

TypeScript schemas can be generated in three styles:

**1. Interface (default):**

```python
ts_schema = manager.get_typescript_schema("User", "1.0.0", style="interface")
```

```typescript
export interface UserV1_0_0 {
  name: string;
  email: string;
  age: number;
}
```

**2. Type alias:**
```python
ts_schema = manager.get_typescript_schema("User", "1.0.0", style="type")
```

```typescript
export type UserV1_0_0 = {
  name: string;
  email: string;
  age: number;
};
```

**3. Zod schema (with runtime validation):**
```python
ts_schema = manager.get_typescript_schema("User", "1.0.0", style="zod")
```

```typescript
import { z } from 'zod';

export const UserV1_0_0Schema = z.object({
  name: z.string(),
  email: z.string(),
  age: z.number().int().gte(0).lte(150),
});

export type UserV1_0_0 = z.infer<typeof UserV1_0_0Schema>;
```

Note how Pydantic constraints are preserved in the Zod schema.

### Using the Generated Schema

The schema is returned as a string, ready to use:

```python
# Get schema as string
ts_schema = manager.get_typescript_schema("User", "1.0.0")

# Write to file
from pathlib import Path
Path("types/user.ts").write_text(ts_schema)

# Or export all schemas at once
manager.dump_typescript_schemas("types/", style="interface")

# Generate Zod schemas
manager.dump_typescript_schemas("schemas/", style="zod")
```

## Type Mapping

### Basic Types

Python types are automatically mapped to TypeScript types:

| Python Type | TypeScript Type |
|-------------|-----------------|
| `str` | `string` |
| `int` | `number` |
| `float` | `number` |
| `bool` | `boolean` |
| `bytes` | `string` |
| `None` | `null` |

```python
@manager.model("BasicTypes", "1.0.0")
class BasicTypesV1(BaseModel):
    name: str          # -> string
    count: int         # -> number
    price: float       # -> number
    active: bool       # -> boolean
    data: bytes        # -> string
```

### Special Types

Special Python types map to TypeScript equivalents:

| Python Type | TypeScript Type | Zod Type |
|-------------|-----------------|----------|
| `datetime` | `string` | `z.string().datetime()` |
| `date` | `string` | `z.string().date()` |
| `time` | `string` | `z.string().time()` |
| `UUID` | `string` | `z.string().uuid()` |
| `Decimal` | `number` | `z.number()` |

```python
from datetime import datetime, date, time
from uuid import UUID
from decimal import Decimal


@manager.model("Event", "1.0.0")
class EventV1(BaseModel):
    event_id: UUID          # -> string
    timestamp: datetime     # -> string
    date: date             # -> string
    time: time             # -> string
    amount: Decimal         # -> number
```

Generated TypeScript:

```typescript
export interface EventV1_0_0 {
  event_id: string;
  timestamp: string;
  date: string;
  time: string;
  amount: number;
}
```

Generated Zod:

```typescript
export const EventV1_0_0Schema = z.object({
  event_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  date: z.string().date(),
  time: z.string().time(),
  amount: z.number(),
});
```

### Collection Types

Lists, sets, tuples, and dictionaries are supported:

```python
@manager.model("Collections", "1.0.0")
class CollectionsV1(BaseModel):
    tags: list[str]              # -> string[]
    scores: list[int]            # -> number[]
    unique_ids: set[str]         # -> string[]
    metadata: dict[str, str]     # -> Record<string, string>
    counts: dict[str, int]       # -> Record<string, number>
    coordinates: tuple[float, float]  # -> [number, number]
```

Generated TypeScript:

```typescript
export interface CollectionsV1_0_0 {
  tags: string[];
  scores: number[];
  unique_ids: string[];
  metadata: Record<string, string>;
  counts: Record<string, number>;
  coordinates: [number, number];
}
```

### Nested Collections

Complex nested structures are fully supported:

```python
@manager.model("Complex", "1.0.0")
class ComplexV1(BaseModel):
    matrix: list[list[int]]                    # -> number[][]
    nested_dict: dict[str, dict[str, str]]    # -> Record<string, Record<string, string>>
    records: list[dict[str, int]]             # -> Record<string, number>[]
    groups: dict[str, list[str]]              # -> Record<string, string[]>
```

Generated TypeScript:

```typescript
export interface ComplexV1_0_0 {
  matrix: number[][];
  nested_dict: Record<string, Record<string, string>>;
  records: Record<string, number>[];
  groups: Record<string, string[]>;
}
```

### Optional Fields

**Fields with defaults:**

Fields with default values become optional in TypeScript:

```python
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str                      # Required
    email: str = "unknown"         # Optional (has default)
    age: int | None = None         # Optional (nullable)
```

Generated TypeScript:

```typescript
export interface UserV1_0_0 {
  name: string;
  email?: string;
  age?: number;
}
```

**Nullable vs optional:**

TypeScript distinguishes between nullable and optional fields:
```python
@manager.model("Model", "1.0.0")
class ModelV1(BaseModel):
    required_nullable: str | None   # Required, can be null
    optional_with_default: str = "default"  # Optional, cannot be null
    optional_nullable: str | None = None    # Optional, treated as omittable
```

Generated TypeScript:
```typescript
export interface ModelV1_0_0 {
  required_nullable: string | null;  // Must be provided
  optional_with_default?: string;    // Can be omitted
  optional_nullable?: string;         // Can be omitted (null treated as omittable)
}
```

!!! note
    Fields with `| None` and a `None` default are generated as optional (`?`)
    rather than explicitly nullable in TypeScript, following the common
    pattern where such fields can simply be omitted by frontend clients.

### Enum Types

Python Enums map to TypeScript union types or enums:

```python
from enum import Enum


class Status(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


@manager.model("Task", "1.0.0")
class TaskV1(BaseModel):
    name: str
    status: Status
```

Generated TypeScript (default: union):

```typescript
export interface TaskV1_0_0 {
  name: string;
  status: 'pending' | 'active' | 'completed';
}
```

Generated Zod:

```typescript
export const TaskV1_0_0Schema = z.object({
  name: z.string(),
  status: z.enum(['pending', 'active', 'completed']),
});
```

**Configuration:** You can generate TypeScript enums instead:

```python
from pyrmute.typescript_types import TypeScriptConfig

config = TypeScriptConfig(enum_style="enum")
ts_schema = manager.get_typescript_schema(
    "Task", "1.0.0",
    style="interface",
    config=config
)
```

```typescript
enum Status {
  PENDING = 'pending',
  ACTIVE = 'active',
  COMPLETED = 'completed'
}

export interface TaskV1_0_0 {
  name: string;
  status: Status;
}
```

### Discriminated Unions

Discriminated unions use a literal field to distinguish between different
types, enabling TypeScript's type narrowing:

```python
from typing import Annotated, Literal
from pydantic import Field

@manager.model("ClickEvent", "1.0.0")
class ClickEventV1(BaseModel):
    event_type: Literal["click"] = "click"
    element_id: str = Field(alias="elementId")
    x: int
    y: int

@manager.model("ViewEvent", "1.0.0")
class ViewEventV1(BaseModel):
    event_type: Literal["view"] = "view"
    page_url: str = Field(alias="pageUrl")
    duration: int

@manager.model("EventContainer", "1.0.0")
class EventContainerV1(BaseModel):
    event: Annotated[
        ClickEventV1 | ViewEventV1,
        Field(discriminator="event_type")
    ]
```

Generated TypeScript:
```typescript
export interface ClickEventV1_0_0 {
  event_type: 'click';  // Required, not optional
  elementId: string;
  x: number;
  y: number;
}

export interface ViewEventV1_0_0 {
  event_type: 'view';  // Required, not optional
  pageUrl: string;
  duration: number;
}

export interface EventContainerV1_0_0 {
  event: ClickEventV1_0_0 | ViewEventV1_0_0;
}
```

**Type narrowing in TypeScript:**

```typescript
function handleEvent(container: EventContainerV1_0_0) {
  switch (container.event.event_type) {
    case 'click':
      // TypeScript knows this is ClickEventV1_0_0
      console.log(`Clicked at ${container.event.x}, ${container.event.y}`);
      break;
    case 'view':
      // TypeScript knows this is ViewEventV1_0_0
      console.log(`Viewed ${container.event.pageUrl}`);
      break;
  }
}
```

!!! important
    Discriminator fields with literal defaults (like `event_type:
    Literal["click"] = "click"`) are correctly generated as required fields
    without the `?` optional marker, even though they have defaults. This is
    because discriminators must always be present for type narrowing to work
    properly.

**Using enum values as discriminators:**

You can also use enum members in discriminated unions:

```python
from enum import Enum

class PaymentMethod(str, Enum):
    CARD = "card"
    BANK = "bank"
    CRYPTO = "crypto"

@manager.model("CardPayment", "1.0.0")
class CardPaymentV1(BaseModel):
    method: Literal[PaymentMethod.CARD] = PaymentMethod.CARD
    card_number: str = Field(alias="cardNumber")
    cvv: str

@manager.model("BankPayment", "1.0.0")
class BankPaymentV1(BaseModel):
    method: Literal[PaymentMethod.BANK] = PaymentMethod.BANK
    account_number: str = Field(alias="accountNumber")
    routing_number: str = Field(alias="routingNumber")

@manager.model("Payment", "1.0.0")
class PaymentV1(BaseModel):
    payment: Annotated[
        CardPaymentV1 | BankPaymentV1,
        Field(discriminator="method")
    ]
```

Generated TypeScript:
```typescript
export interface CardPaymentV1_0_0 {
  method: 'card';  // Uses enum value, not 'PaymentMethod.CARD'
  cardNumber: string;
  cvv: string;
}

export interface BankPaymentV1_0_0 {
  method: 'bank';
  accountNumber: string;
  routingNumber: string;
}

export interface PaymentV1_0_0 {
  payment: CardPaymentV1_0_0 | BankPaymentV1_0_0;
}
```

**With Zod schemas:**

Discriminated unions work with Zod's discriminated union validation:

```typescript
import { z } from 'zod';

export const ClickEventV1_0_0Schema = z.object({
  event_type: z.literal('click'),
  elementId: z.string(),
  x: z.number().int(),
  y: z.number().int(),
});

export const ViewEventV1_0_0Schema = z.object({
  event_type: z.literal('view'),
  pageUrl: z.string(),
  duration: z.number().int(),
});

export const EventContainerV1_0_0Schema = z.object({
  event: z.union([ClickEventV1_0_0Schema, ViewEventV1_0_0Schema]),
});

// Runtime validation with type narrowing
const event = EventContainerV1_0_0Schema.parse(data);
if (event.event.event_type === 'click') {
  // TypeScript knows event.event is ClickEventV1_0_0
  console.log(event.event.elementId);
}
```

### Literal Types

Python `Literal` types map directly to TypeScript:

```python
from typing import Literal


@manager.model("Config", "1.0.0")
class ConfigV1(BaseModel):
    environment: Literal["dev", "staging", "prod"]
    log_level: Literal["debug", "info", "warning", "error"]
```

Generated TypeScript:

```typescript
export interface ConfigV1_0_0 {
  environment: 'dev' | 'staging' | 'prod';
  log_level: 'debug' | 'info' | 'warning' | 'error';
}
```

### Union Types

Union types are preserved:

```python
@manager.model("Flexible", "1.0.0")
class FlexibleV1(BaseModel):
    value: str | int              # -> string | number
    data: str | dict[str, str]    # -> string | Record<string, string>
```

Generated TypeScript:

```typescript
export interface FlexibleV1_0_0 {
  value: string | number;
  data: string | Record<string, string>;
}
```

### Nested Models

Pydantic models reference other TypeScript types:

```python
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str
    zip_code: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1
```

Generated TypeScript:

```typescript
export interface AddressV1_0_0 {
  street: string;
  city: string;
  zip_code: string;
}

export interface UserV1_0_0 {
  name: string;
  address: AddressV1_0_0;
}
```

### Recursive Models

Self-referential models use forward references:

```python
@manager.model("TreeNode", "1.0.0")
class TreeNodeV1(BaseModel):
    value: int
    children: list["TreeNodeV1"] = []
```

Generated TypeScript:

```typescript
export interface TreeNodeV1_0_0 {
  value: number;
  children?: TreeNodeV1_0_0[];
}
```

### Generic Models

Generic Pydantic models become generic TypeScript types:

```python
from typing import Generic, TypeVar

T = TypeVar("T")


@manager.model("ApiResponse", "1.0.0")
class ApiResponseV1(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    data: T | None = None
    error: str | None = None
    success: bool = True
```

Generated TypeScript:

```typescript
/**
 * Generic API response wrapper.
 */
export interface ApiResponseV1_0_0<T> {
  data?: T;
  error?: string;
  success?: boolean;
}
```

Multiple type parameters:

```python
K = TypeVar("K")
V = TypeVar("V")


@manager.model("KeyValuePair", "1.0.0")
class KeyValuePairV1(BaseModel, Generic[K, V]):
    key: K
    value: V
```

```typescript
export interface KeyValuePairV1_0_0<K, V> {
  key: K;
  value: V;
}
```

### Computed Fields

Computed fields are included by default:

```python
from pydantic import computed_field


@manager.model("Person", "1.0.0")
class PersonV1(BaseModel):
    first_name: str
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
```

Generated TypeScript:

```typescript
export interface PersonV1_0_0 {
  first_name: string;
  last_name: string;
  full_name: string;
}
```

**Mark as readonly:**

```python
config = TypeScriptConfig(mark_computed_readonly=True)
ts_schema = manager.get_typescript_schema(
    "Person", "1.0.0",
    config=config
)
```

```typescript
export interface PersonV1_0_0 {
  first_name: string;
  last_name: string;
  readonly full_name: string;
}
```

**Exclude computed fields:**

```python
config = TypeScriptConfig(include_computed_fields=False)
```

## Configuration Options

Customize TypeScript generation with `TypeScriptConfig`:

```python
from pyrmute.typescript_types import TypeScriptConfig

config = TypeScriptConfig(
    date_format="timestamp",           # "iso" or "timestamp"
    enum_style="union",                # "union" or "enum"
    include_computed_fields=True,      # Include @computed_field
    mark_computed_readonly=False,      # Mark computed as readonly
)

ts_schema = manager.get_typescript_schema(
    "User", "1.0.0",
    style="interface",
    config=config
)
```

### Date Format Options

**ISO strings (default):**

```python
config = TypeScriptConfig(date_format="iso")
```

```typescript
// datetime, date, time -> string
timestamp: string;
```

**Timestamps:**
```python
config = TypeScriptConfig(date_format="timestamp")
```

```typescript
// datetime, date -> number (milliseconds since epoch)
timestamp: number;
```

### Enum Style Options

**Union types (default):**
```python
config = TypeScriptConfig(enum_style="union")
```

```typescript
status: 'pending' | 'active' | 'completed';
```

**Enum declarations:**
```python
config = TypeScriptConfig(enum_style="enum")
```

```typescript
enum Status {
  PENDING = 'pending',
  ACTIVE = 'active',
  COMPLETED = 'completed'
}

status: Status;
```

## Validation Constraints

Pydantic validation constraints are preserved in Zod schemas:

```python
from pydantic import Field


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(ge=0, le=150)
    score: float = Field(gt=0, lt=100)
```

Generated Zod:

```typescript
export const UserV1_0_0Schema = z.object({
  username: z.string().min(3).max(20),
  email: z.string().regex(/^[\w\.-]+@[\w\.-]+\.\w+$/),
  age: z.number().int().gte(0).lte(150),
  score: z.number().gt(0).lt(100),
});
```

**Supported constraints:**

| Pydantic | Zod |
|----------|-----|
| `min_length` | `.min()` |
| `max_length` | `.max()` |
| `pattern` | `.regex()` |
| `ge` (greater than or equal) | `.gte()` |
| `gt` (greater than) | `.gt()` |
| `le` (less than or equal) | `.lte()` |
| `lt` (less than) | `.lt()` |

### Extra Fields

Models with `extra="allow"` generate index signatures allowing additional
properties:

```python
from pydantic import ConfigDict

@manager.model("FlexibleConfig", "1.0.0")
class FlexibleConfigV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    value: int
```

Generated TypeScript:
```typescript
export interface FlexibleConfigV1_0_0 {
  name: string;
  value: number;
  [key: string]: any;  // Allows additional properties
}
```

Generated Zod:
```typescript
export const FlexibleConfigV1_0_0Schema = z.object({
  name: z.string(),
  value: z.number().int(),
}).passthrough();  // Allows additional properties
```

## Exporting Schemas

### Export All Schemas

Export all registered models to a directory:

```python
# Export interfaces
manager.dump_typescript_schemas("frontend/types/", style="interface")

# Export Zod schemas
manager.dump_typescript_schemas("frontend/schemas/", style="zod")

# Export both
manager.dump_typescript_schemas("frontend/types/", style="interface")
manager.dump_typescript_schemas("frontend/schemas/", style="zod")
```

Directory structure:

```
frontend/
├── types/
│   ├── User_v1_0_0.ts
│   ├── Order_v1_0_0.ts
│   └── Product_v1_0_0.ts
└── schemas/
    ├── User_v1_0_0.ts
    ├── Order_v1_0_0.ts
    └── Product_v1_0_0.ts
```

### Export Single Schema

```python
# Get as string
schema = manager.get_typescript_schema("User", "1.0.0")

# Write to file
from pathlib import Path
Path("types/user.ts").write_text(schema)
```

### Integration with Build Process

```python
# scripts/generate_types.py
from pyrmute import ModelManager
from pathlib import Path


def generate_typescript_types() -> None:
    """Generate TypeScript types for frontend."""
    manager = ModelManager()

    # Import all models
    from app import models  # Your models module

    # Export interfaces
    output_dir = Path("frontend/src/types/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    manager.dump_typescript_schemas(
        str(output_dir),
        style="interface"
    )

    print(f"Generated {len(manager.registry)} TypeScript interfaces")


if __name__ == "__main__":
    generate_typescript_types()
```

Run during development:

```bash
# Add to package.json scripts
{
  "scripts": {
    "generate-types": "python scripts/generate_types.py",
    "dev": "npm run generate-types && vite"
  }
}
```

## Schema Evolution Best Practices

### Backward Compatible Changes

Add new optional fields:

```python
# Version 1
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    email: str


# Version 2: Add optional fields (backward compatible)
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    email: str
    phone: str | None = None      # New optional field
    address: str | None = None    # New optional field
```

Generated TypeScript (v2):

```typescript
export interface UserV2_0_0 {
  name: string;
  email: string;
  phone?: string;     // New field
  address?: string;   // New field
}
```

Old frontend code continues to work with new API responses.

### Breaking Changes

Change required fields (new version):

```python
# Version 2: Breaking change
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    id: str           # New required field
    name: str
    email: str
```

Frontend must be updated to handle new field.

### Deprecation

Mark fields as deprecated with comments:

```python
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    email: str
    username: str | None = Field(
        None,
        description="DEPRECATED: Use email instead"
    )
```

```typescript
export interface UserV2_0_0 {
  name: string;
  email: string;
  /** DEPRECATED: Use email instead */
  username?: string;
}
```

## Real-World Examples

### REST API Client

```python
from datetime import datetime
from uuid import UUID


@manager.model("CreateUserRequest", "1.0.0")
class CreateUserRequestV1(BaseModel):
    """Request to create a new user."""
    name: str
    email: str
    password: str


@manager.model("UserResponse", "1.0.0")
class UserResponseV1(BaseModel):
    """User data returned from API."""
    id: UUID
    name: str
    email: str
    created_at: datetime
    updated_at: datetime


# Export for frontend
manager.dump_typescript_schemas("frontend/src/types/", style="interface")
manager.dump_typescript_schemas("frontend/src/schemas/", style="zod")
```

Frontend usage:

```typescript
import type { CreateUserRequestV1_0_0, UserResponseV1_0_0 } from './types';
import { UserResponseV1_0_0Schema } from './schemas';

async function createUser(data: CreateUserRequestV1_0_0): Promise<UserResponseV1_0_0> {
  const response = await fetch('/api/users', {
    method: 'POST',
    body: JSON.stringify(data),
  });

  const json = await response.json();

  // Runtime validation
  return UserResponseV1_0_0Schema.parse(json);
}
```

### Form Validation

```python
from pydantic import Field


@manager.model("LoginForm", "1.0.0")
class LoginFormV1(BaseModel):
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(min_length=8, max_length=100)
    remember_me: bool = False
```

Frontend with Zod:

```typescript
import { LoginFormV1_0_0Schema } from './schemas';

// Integrate with React Hook Form
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';

function LoginForm() {
  const form = useForm({
    resolver: zodResolver(LoginFormV1_0_0Schema),
  });

  // Form automatically validates with backend rules
}
```

### WebSocket Events

```python
from typing import Literal


@manager.model("ChatMessage", "1.0.0")
class ChatMessageV1(BaseModel):
    type: Literal["message"]
    user_id: str
    content: str
    timestamp: datetime


@manager.model("UserJoined", "1.0.0")
class UserJoinedV1(BaseModel):
    type: Literal["user_joined"]
    user_id: str
    username: str


@manager.model("UserLeft", "1.0.0")
class UserLeftV1(BaseModel):
    type: Literal["user_left"]
    user_id: str


# Union type for all events
from typing import Union
WebSocketEvent = Union[ChatMessageV1, UserJoinedV1, UserLeftV1]
```

Frontend usage:

```typescript
type WebSocketEvent =
  | ChatMessageV1_0_0
  | UserJoinedV1_0_0
  | UserLeftV1_0_0;

function handleMessage(event: WebSocketEvent) {
  // TypeScript discriminated union
  switch (event.type) {
    case 'message':
      // event is ChatMessageV1_0_0
      console.log(event.content);
      break;
    case 'user_joined':
      // event is UserJoinedV1_0_0
      console.log(`${event.username} joined`);
      break;
    case 'user_left':
      // event is UserLeftV1_0_0
      console.log(`User ${event.user_id} left`);
      break;
  }
}
```

### API Response Wrappers

```python
from typing import Generic, TypeVar

T = TypeVar("T")


@manager.model("ApiResponse", "1.0.0")
class ApiResponseV1(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    data: T | None = None
    error: str | None = None
    success: bool
    timestamp: datetime
```

Frontend usage:

```typescript
// Type-safe API responses
type UserListResponse = ApiResponseV1_0_0<UserResponseV1_0_0[]>;
type UserDetailResponse = ApiResponseV1_0_0<UserResponseV1_0_0>;

async function getUsers(): Promise<UserListResponse> {
  const response = await fetch('/api/users');
  return response.json();
}
```

## Common Patterns

### Pagination

```python
@manager.model("PaginationParams", "1.0.0")
class PaginationParamsV1(BaseModel):
    page: int = Field(ge=1, default=1)
    page_size: int = Field(ge=1, le=100, default=20)


@manager.model("PaginatedResponse", "1.0.0")
class PaginatedResponseV1(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool
```

### Error Responses

```python
from enum import Enum


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    SERVER_ERROR = "server_error"


@manager.model("ErrorResponse", "1.0.0")
class ErrorResponseV1(BaseModel):
    error_code: ErrorCode
    message: str
    details: dict[str, str] | None = None
    timestamp: datetime
```

### Filters and Search

```python
@manager.model("UserFilter", "1.0.0")
class UserFilterV1(BaseModel):
    search: str | None = None
    role: str | None = None
    active: bool | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
```

## Testing

### Validate Generated Types

```python
import subprocess
from pathlib import Path


def test_typescript_validity() -> None:
    """Test that generated TypeScript is syntactically valid."""
    # Generate schemas
    output_dir = Path("test_types")
    manager.dump_typescript_schemas(str(output_dir), style="interface")

    # Run TypeScript compiler in check mode
    result = subprocess.run(
        ["tsc", "--noEmit", "--strict", str(output_dir / "*.ts")],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"TypeScript errors: {result.stderr}"


def test_zod_schemas():
    """Test that Zod schemas are valid."""
    output_dir = Path("test_schemas")
    manager.dump_typescript_schemas(str(output_dir), style="zod")

    # TypeScript compiler validates Zod schemas
    result = subprocess.run(
        ["tsc", "--noEmit", str(output_dir / "*.ts")],
        capture_output=True
    )

    assert result.returncode == 0
```

### Runtime Validation

```python
def test_zod_validation() -> None:
    """Test that Zod schemas validate correctly."""
    # This would be a Node.js test
    # test/validation.test.ts:
    """
    import { UserV1_0_0Schema } from '../types';

    describe('User validation', () => {
      it('validates correct data', () => {
        const valid = {
          name: 'Alice',
          email: 'alice@example.com',
          age: 30
        };

        expect(() => UserV1_0_0Schema.parse(valid)).not.toThrow();
      });

      it('rejects invalid data', () => {
        const invalid = {
          name: 'Alice',
          email: 'not-an-email',
          age: -5
        };

        expect(() => UserV1_0_0Schema.parse(invalid)).toThrow();
      });
    });
    """
```

## Troubleshooting

### Import Errors

If Zod types aren't found:

```bash
# Ensure zod is installed
npm install zod

# Or with pnpm
pnpm install zod

# Or with yarn
yarn add zod
```

### TypeScript Compilation Errors

Check generated syntax:

```bash
# Validate TypeScript
tsc --noEmit types/*.ts

# Check for common issues
tsc --strict types/*.ts
```

### Type Mismatches

If TypeScript complains about types:

```typescript
// Use type assertion
const user = data as UserV1_0_0;

// Or validate with Zod
const user = UserV1_0_0Schema.parse(data);
```

### Circular Dependencies

For recursive types, TypeScript may need help:

```typescript
// If TypeScript can't resolve recursive types
export interface TreeNodeV1_0_0 {
  value: number;
  children?: TreeNodeV1_0_0[]; // May need explicit annotation
}
```

## Comparison with JSON Schema

| Feature | JSON Schema | TypeScript |
|---------|-------------|------------|
| Generation | `get_schema()` | `get_typescript_schema()` |
| Export | `dump_schemas()` | `dump_typescript_schemas()` |
| Syntax | JSON | TypeScript |
| Runtime validation | Yes | Only with Zod |
| Compile-time | No | Yes |
| IDE support | Limited | Excellent |
| Use case | API documentation | Frontend development |

## Best Practices

1. **Use Zod for runtime validation** - Don't trust API data
2. **Generate types in CI/CD** - Keep frontend and backend in sync
3. **Version control generated files** - Track type changes in Git
4. **Use interfaces for most cases** - Better for extending
5. **Mark computed fields readonly** - Clarify intent
6. **Document breaking changes** - Use JSDoc comments
7. **Test TypeScript compilation** - Validate in CI
8. **Use generics for reusable types** - Reduce duplication

## Next Steps

**Related topics:**

- [Schema Generation](schema-generation.md) - JSON Schema generation
- [Protocol Buffer Generation](protobuf-schemas.md) - Protobuf schemas

**External resources:**

- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Zod Documentation](https://zod.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)

**API Reference:**

- [`TypeScriptSchemaGenerator`](../reference/typescript-schema.md)
- [`ModelManager`](../reference/model-manager.md)
