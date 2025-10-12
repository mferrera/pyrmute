# Nested Models

pyrmute automatically handles migrations for nested Pydantic models. This
guide covers how nested model migrations work, common patterns, and best
practices.

## Basic Nested Models

When a model contains another model, pyrmute migrates both:

```python
from pydantic import BaseModel
from pyrmute import ModelManager, ModelData

manager = ModelManager()


# Register nested model
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str
    postal_code: str


# Register parent model
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2  # Uses AddressV2


# Define migrations
@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    return {**data, "postal_code": "00000"}


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    # pyrmute automatically migrates nested address
    return data


# Migrate user with nested address
old_user = {
    "name": "Alice",
    "address": {"street": "123 Main St", "city": "NYC"}
}

new_user = manager.migrate(old_user, "User", "1.0.0", "2.0.0")
print(new_user)
# UserV2(
#   name="Alice",
#   address=AddressV2(
#     street="123 Main St",
#     city="NYC",
#     postal_code="00000"  # Added by address migration
#   )
# )
```

**How it works:**

1. User migration runs first
2. pyrmute detects `address` is a nested model (AddressV1 -> AddressV2)
3. Address migration runs automatically
4. Result is validated against UserV2

## Automatic Nested Migration

You don't need to manually migrate nested models in your migration functions:

```python
# ❌ DON'T DO THIS - Unnecessary
@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user_manual(data: ModelData) -> ModelData:
    # Manually migrating address (not needed!)
    address_data = data["address"]
    migrated_address = manager.migrate_data(
        address_data,
        "Address",
        "1.0.0",
        "2.0.0"
    )
    return {
        **data,
        "address": migrated_address
    }


# ✅ DO THIS - Let pyrmute handle it
@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user_auto(data: ModelData) -> ModelData:
    # pyrmute automatically handles nested address
    return data
```

pyrmute automatically:

- Detects nested Pydantic models
- Determines source and target versions
- Applies appropriate migrations
- Validates the result

## Lists of Nested Models

pyrmute handles lists of models automatically:

```python
@manager.model("Item", "1.0.0")
class ItemV1(BaseModel):
    name: str
    price: float


@manager.model("Item", "2.0.0")
class ItemV2(BaseModel):
    name: str
    price: float
    currency: str


@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    items: list[ItemV1]


@manager.model("Order", "2.0.0")
class OrderV2(BaseModel):
    order_id: str
    items: list[ItemV2]


@manager.migration("Item", "1.0.0", "2.0.0")
def migrate_item(data: ModelData) -> ModelData:
    return {**data, "currency": "USD"}


@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    # pyrmute migrates each item in the list
    return data


# Migrate order with multiple items
old_order = {
    "order_id": "ORD-123",
    "items": [
        {"name": "Widget", "price": 9.99},
        {"name": "Gadget", "price": 19.99},
    ]
}

new_order = manager.migrate(old_order, "Order", "1.0.0", "2.0.0")
print(new_order)
# OrderV2(
#   order_id="ORD-123",
#   items=[
#     ItemV2(name="Widget", price=9.99, currency="USD"),
#     ItemV2(name="Gadget", price=19.99, currency="USD")
#   ]
# )
```

## Deeply Nested Models

Migrations work recursively for deeply nested structures:

```python
@manager.model("Country", "1.0.0")
class CountryV1(BaseModel):
    code: str
    name: str


@manager.model("Country", "2.0.0")
class CountryV2(BaseModel):
    code: str
    name: str
    region: str


@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    country: CountryV1


@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    country: CountryV2


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2


# Define migrations for each level
@manager.migration("Country", "1.0.0", "2.0.0")
def migrate_country(data: ModelData) -> ModelData:
    return {**data, "region": "Unknown"}


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    return data  # Country migration handled automatically


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    return data  # Address and Country migrations handled automatically


# Migrate deeply nested structure
old_user = {
    "name": "Alice",
    "address": {
        "street": "123 Main St",
        "country": {"code": "US", "name": "United States"}
    }
}

new_user = manager.migrate(old_user, "User", "1.0.0", "2.0.0")
print(new_user)
# UserV2(
#   name="Alice",
#   address=AddressV2(
#     street="123 Main St",
#     country=CountryV2(
#       code="US",
#       name="United States",
#       region="Unknown"  # Added by country migration
#     )
#   )
# )
```

**Key point:** Each nested model is migrated independently, but pyrmute
handles the coordination automatically.

!!! note "Recursive Models"
    pyrmute does not currently detect recursive models.

## Optional Nested Models

Handle optional nested models with proper defaults:

```python
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str
    postal_code: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1 | None = None


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2 | None = None


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    return {**data, "postal_code": "00000"}


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    # No special handling needed for None
    return data


# With address
user1 = manager.migrate(
    {"name": "Alice", "address": {"street": "123 Main", "city": "NYC"}},
    "User",
    "1.0.0",
    "2.0.0"
)
# Result: UserV2(name="Alice", address=AddressV2(...))

# Without address
user2 = manager.migrate(
    {"name": "Bob"},
    "User",
    "1.0.0",
    "2.0.0"
)
# Result: UserV2(name="Bob", address=None)
```

## Version Synchronization

### Synchronized Versions (Recommended)

Keep parent and child versions synchronized:

```python
# Both at v1.0.0
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


# Both upgrade to v2.0.0 together
@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    return {**data, "city": "Unknown"}


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    return data
```

**Benefits:**

- Simpler to reason about
- Fewer migration paths
- Easier testing

### Independent Versions (Advanced)

Allow nested models to version independently:

```python
# Address v1.0.0
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str


# User v1.0.0 uses Address v1.0.0
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


# Address evolves to v2.0.0
@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    return {**data, "city": "Unknown"}


# User v2.0.0 still uses Address v1.0.0
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    email: str
    address: AddressV1  # Still v1!


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    return {**data, "email": "unknown@example.com"}


# Later, User v3.0.0 upgrades to Address v2.0.0
@manager.model("User", "3.0.0")
class UserV3(BaseModel):
    name: str
    email: str
    address: AddressV2  # Now v2!


@manager.migration("User", "2.0.0", "3.0.0")
def migrate_user_v2_to_v3(data: ModelData) -> ModelData:
    # Address automatically migrates from v1 to v2
    return data
```

**Use cases:**

- Shared models across different domains
- Independent release cycles
- Gradual rollout of changes

## Nested Model Chains

When both parent and child have multiple versions, pyrmute chains migrations:

```python
# Address versions
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str


@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str


@manager.model("Address", "3.0.0")
class AddressV3(BaseModel):
    street: str
    city: str
    postal_code: str


# User versions
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2


@manager.model("User", "3.0.0")
class UserV3(BaseModel):
    name: str
    address: AddressV3


# Define all migrations
@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address_v1_to_v2(data: ModelData) -> ModelData:
    return {**data, "city": "Unknown"}


@manager.migration("Address", "2.0.0", "3.0.0")
def migrate_address_v2_to_v3(data: ModelData) -> ModelData:
    return {**data, "postal_code": "00000"}


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user_v1_to_v2(data: ModelData) -> ModelData:
    return data


@manager.migration("User", "2.0.0", "3.0.0")
def migrate_user_v2_to_v3(data: ModelData) -> ModelData:
    return data


# Migrate from v1.0.0 to v3.0.0
old_user = {
    "name": "Alice",
    "address": {"street": "123 Main St"}
}

new_user = manager.migrate(old_user, "User", "1.0.0", "3.0.0")
print(new_user)
# UserV3(
#   name="Alice",
#   address=AddressV3(
#     street="123 Main St",
#     city="Unknown",      # From Address v1->v2
#     postal_code="00000"  # From Address v2->v3
#   )
# )
```

**Migration order:**

1. User v1->v2 migration
2. Address v1->v2 migration (for nested address)
3. User v2->v3 migration
4. Address v2->v3 migration (for nested address)

## Discriminated Unions

pyrmute supports discriminated unions for polymorphic nested models:

```python
from typing import Literal, Union
from pydantic import Field


# Payment method types
@manager.model("CreditCard", "1.0.0")
class CreditCardV1(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str
    expiry: str


@manager.model("PayPal", "1.0.0")
class PayPalV1(BaseModel):
    type: Literal["paypal"] = "paypal"
    email: str


# Order with discriminated union
@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV1, PayPalV1] = Field(discriminator="type")


# Evolve credit card model
@manager.model("CreditCard", "2.0.0")
class CreditCardV2(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str
    expiry: str
    cvv: str  # New field


@manager.model("PayPal", "2.0.0")
class PayPalV2(BaseModel):
    type: Literal["paypal"] = "paypal"
    email: str
    verified: bool  # New field


@manager.model("Order", "2.0.0")
class OrderV2(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV2, PayPalV2] = Field(discriminator="type")


# Migrations for payment methods
@manager.migration("CreditCard", "1.0.0", "2.0.0")
def migrate_credit_card(data: ModelData) -> ModelData:
    return {**data, "cvv": "000"}


@manager.migration("PayPal", "1.0.0", "2.0.0")
def migrate_paypal(data: ModelData) -> ModelData:
    return {**data, "verified": False}


@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    # pyrmute uses discriminator to determine which migration to run
    return data


# Migrate order with credit card
order1 = manager.migrate(
    {
        "order_id": "ORD-1",
        "payment_method": {
            "type": "credit_card",
            "card_number": "1234",
            "expiry": "12/25"
        }
    },
    "Order",
    "1.0.0",
    "2.0.0"
)
# CreditCard migration runs automatically

# Migrate order with PayPal
order2 = manager.migrate(
    {
        "order_id": "ORD-2",
        "payment_method": {
            "type": "paypal",
            "email": "user@example.com"
        }
    },
    "Order",
    "1.0.0",
    "2.0.0"
)
# PayPal migration runs automatically
```

**How it works:**

1. pyrmute reads the discriminator field (`type`)
2. Determines which model type the data represents
3. Applies the appropriate migration
4. Validates against the correct target model

## Handling Field Aliases

pyrmute respects Pydantic field aliases in nested models:

```python
from pydantic import Field


@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street_name: str = Field(alias="streetName")
    city_name: str = Field(alias="cityName")


@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street_name: str = Field(alias="streetName")
    city_name: str = Field(alias="cityName")
    postal_code: str = Field(alias="postalCode")


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    address: AddressV2


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    return {**data, "postalCode": "00000"}


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    return data


# Data with aliases
old_user = {
    "name": "Alice",
    "address": {
        "streetName": "123 Main St",  # Using alias
        "cityName": "NYC"             # Using alias
    }
}

new_user = manager.migrate(old_user, "User", "1.0.0", "2.0.0")
# Aliases are preserved through migration
```

## Testing Nested Migrations

Test nested model migrations thoroughly:

```python
def test_nested_model_migration() -> None:
    """Test that nested models migrate correctly."""
    results = manager.test_migration(
        "User",
        "1.0.0",
        "2.0.0",
        test_cases=[
            # With nested address
            (
                {
                    "name": "Alice",
                    "address": {"street": "123 Main", "city": "NYC"}
                },
                {
                    "name": "Alice",
                    "address": {
                        "street": "123 Main",
                        "city": "NYC",
                        "postal_code": "00000"
                    }
                }
            ),
            # Without nested address (None)
            (
                {"name": "Bob"},
                {"name": "Bob", "address": None}
            ),
        ]
    )
    results.assert_all_passed()


def test_list_of_nested_models() -> None:
    """Test migration of lists of nested models."""
    results = manager.test_migration(
        "Order",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {
                    "order_id": "ORD-1",
                    "items": [
                        {"name": "Widget", "price": 9.99},
                        {"name": "Gadget", "price": 19.99}
                    ]
                },
                {
                    "order_id": "ORD-1",
                    "items": [
                        {"name": "Widget", "price": 9.99, "currency": "USD"},
                        {"name": "Gadget", "price": 19.99, "currency": "USD"}
                    ]
                }
            ),
            # Empty list
            (
                {"order_id": "ORD-2", "items": []},
                {"order_id": "ORD-2", "items": []}
            ),
        ]
    )
    results.assert_all_passed()
```

## Common Patterns

### Shared Models Across Domains

```python
# Shared address model used by multiple domains
@manager.model("Address", "1.0.0")
class AddressV1(BaseModel):
    street: str
    city: str


# Used by User
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1


# Used by Company
@manager.model("Company", "1.0.0")
class CompanyV1(BaseModel):
    name: str
    headquarters: AddressV1  # Same address model


# When Address evolves, both User and Company automatically benefit
@manager.model("Address", "2.0.0")
class AddressV2(BaseModel):
    street: str
    city: str
    postal_code: str


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData): -> ModelData
    return {**data, "postal_code": "00000"}
```

### Embedding vs. Referencing

```python
# Embedded model (default)
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1  # Address data embedded in user


# Referenced model (with enable_ref)
@manager.model("Address", "1.0.0", enable_ref=True)
class AddressV1(BaseModel):
    street: str
    city: str


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1  # Can use $ref in schemas


# When dumping schemas with separate_definitions=True:
manager.dump_schemas("schemas/", separate_definitions=True)
# Creates: User_v1.0.0.json (with $ref to Address)
#          Address_v1.0.0.json (separate file)
```

### Flattening and Nesting

```python
# Flatten: Nested -> Flat
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    name: str
    address: AddressV1  # Nested


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    street: str  # Flattened
    city: str    # Flattened


@manager.migration("User", "1.0.0", "2.0.0")
def flatten_address(data: ModelData) -> ModelData:
    address = data.get("address", {})
    return {
        "name": data["name"],
        "street": address.get("street", ""),
        "city": address.get("city", "")
    }


# Nest: Flat → Nested
@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    name: str
    street: str
    city: str


@manager.model("User", "3.0.0")
class UserV3(BaseModel):
    name: str
    address: AddressV2  # Nested


@manager.migration("User", "2.0.0", "3.0.0")
def nest_address(data: ModelData) -> ModelData:
    return {
        "name": data["name"],
        "address": {
            "street": data["street"],
            "city": data["city"]
        }
    }
```

## Performance Considerations

Nested migrations can multiply processing time:

```python
# Order with 100 items, each migrating through 3 versions
# Results in 300 individual item migrations!
order = manager.migrate(
    {
        "order_id": "ORD-1",
        "items": [{"name": f"Item {i}", "price": 9.99} for i in range(100)]
    },
    "Order",
    "1.0.0",
    "4.0.0"
)
```

**Optimization strategies:**

1. Minimize number of version hops
2. Use direct migrations (skip intermediate versions)
3. Keep nested models simple
4. Consider flattening deeply nested structures

## Best Practices

1. **Keep nesting shallow** - Avoid deeply nested structures when possible
2. **Version synchronously** - Upgrade parent and child together when practical
3. **Test all paths** - Test with and without nested data
4. **Document dependencies** - Note which models are nested in others
5. **Use discriminated unions** - For polymorphic nested models
6. **Profile performance** - Measure impact of nested migrations on large datasets

## Troubleshooting

### Nested Model Not Migrating

Check that:

- Both models are registered
- Both migrations are defined
- Field name matches in both versions
- Model is actually a Pydantic BaseModel

### Version Mismatch

```python
# If you see unexpected versions, check your model definitions
@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    address: AddressV1  # Make sure this is the right version


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    address: AddressV2  # And this matches what you expect
```

### Performance Issues

For large datasets with nested models:

- Use batch processing with parallelization
- Profile to find bottlenecks
- Consider denormalizing data
- Cache migration results if possible

## Next Steps

Now that you understand nested model migrations:

**Continue learning:**

- [Discriminated Unions](../advanced/discriminated-unions.md) - Polymorphic
    nested models in detail
- [Batch Processing](batch-processing.md) - Efficiently migrate data with
    nested models
- [Schema Generation](schema-generation.md) - How nested models appear in
    schemas

**Related topics:**

- [Writing Migrations](writing-migrations.md) - Best practices for nested
    transformations
- [Testing Migrations](testing-migrations.md) - Testing nested model
    migrations

**API Reference:**

- [`ModelManager` API](../reference/model-manager.md) - Complete
    `ModelManager` details
- [Exceptions](../reference/exceptions.md) - Exceptions pyrmute raises
- [Types](../reference/types.md) - Type alises exported by pyrmute
