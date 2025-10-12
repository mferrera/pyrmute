# Discriminated Unions

Discriminated unions allow a single field to contain different types of
models, distinguished by a discriminator field. This guide covers how pyrmute
handles polymorphic models, migration patterns, and best practices.

## What are Discriminated Unions?

A discriminated union is a field that can be one of several model types, with
a special field (the discriminator) indicating which type it is:

```python
from typing import Union, Literal
from pydantic import BaseModel, Field
from pyrmute import ModelManager

manager = ModelManager()


# Different payment method types
@manager.model("CreditCard", "1.0.0")
class CreditCardV1(BaseModel):
    type: Literal["credit_card"] = "credit_card"  # Discriminator
    card_number: str
    expiry: str


@manager.model("PayPal", "1.0.0")
class PayPalV1(BaseModel):
    type: Literal["paypal"] = "paypal"  # Discriminator
    email: str


@manager.model("BankTransfer", "1.0.0")
class BankTransferV1(BaseModel):
    type: Literal["bank_transfer"] = "bank_transfer"  # Discriminator
    account_number: str
    routing_number: str


# Order uses discriminated union
@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV1, PayPalV1, BankTransferV1] = Field(
        discriminator="type"
    )


# Create orders with different payment methods
credit_card_order = OrderV1(
    order_id="ORD-1",
    payment_method={
        "type": "credit_card",
        "card_number": "4111111111111111",
        "expiry": "12/25"
    }
)

paypal_order = OrderV1(
    order_id="ORD-2",
    payment_method={
        "type": "paypal",
        "email": "user@example.com"
    }
)
```

**Key components:**

- **Discriminator field** - Field that indicates the type (usually named
    `type`)
- **Literal types** - Each model has a unique literal value
- **Union type** - Combined using `Union[...]` or `|`
- **Field discriminator** - Tells Pydantic which field is the discriminator

## Basic Migration

pyrmute automatically handles discriminated union migrations:

```python
# Evolve CreditCard model
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


@manager.model("BankTransfer", "2.0.0")
class BankTransferV2(BaseModel):
    type: Literal["bank_transfer"] = "bank_transfer"
    account_number: str
    routing_number: str
    swift_code: str  # New field


@manager.model("Order", "2.0.0")
class OrderV2(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV2, PayPalV2, BankTransferV2] = Field(
        discriminator="type"
    )


# Define migrations for each payment type
@manager.migration("CreditCard", "1.0.0", "2.0.0")
def migrate_credit_card(data: ModelData) -> ModelData:
    return {**data, "cvv": "000"}


@manager.migration("PayPal", "1.0.0", "2.0.0")
def migrate_paypal(data: ModelData) -> ModelData:
    return {**data, "verified": False}


@manager.migration("BankTransfer", "1.0.0", "2.0.0")
def migrate_bank_transfer(data: ModelData) -> ModelData:
    return {**data, "swift_code": "UNKNOWN"}


@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    # pyrmute automatically migrates payment_method based on type
    return data


# Migrate order with credit card
old_order = {
    "order_id": "ORD-1",
    "payment_method": {
        "type": "credit_card",
        "card_number": "4111111111111111",
        "expiry": "12/25"
    }
}

new_order = manager.migrate(old_order, "Order", "1.0.0", "2.0.0")
print(new_order.payment_method)
# CreditCardV2(type='credit_card', card_number='...', expiry='12/25', cvv='000')
```

**How it works:**

1. pyrmute reads the discriminator field (`type`)
2. Determines which model type (CreditCard, PayPal, or BankTransfer)
3. Applies the appropriate migration for that type
4. Validates against the correct target model

## Discriminator Field Patterns

### Using 'type' (Recommended)

Most common pattern:

```python
@manager.model("Dog", "1.0.0")
class DogV1(BaseModel):
    type: Literal["dog"] = "dog"
    name: str
    breed: str


@manager.model("Cat", "1.0.0")
class CatV1(BaseModel):
    type: Literal["cat"] = "cat"
    name: str
    indoor: bool


@manager.model("Pet", "1.0.0")
class PetV1(BaseModel):
    owner: str
    animal: Union[DogV1, CatV1] = Field(discriminator="type")
```

!!! tip
    `kind` is a good field name replacement if you wish to avoid overlapping
    with Python's `type` keyword.

### Custom Discriminator Names

Use a different field name:

```python
@manager.model("Admin", "1.0.0")
class AdminV1(BaseModel):
    role: Literal["admin"] = "admin"  # Custom discriminator
    permissions: list[str]


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    role: Literal["user"] = "user"
    quota: int


@manager.model("Account", "1.0.0")
class AccountV1(BaseModel):
    username: str
    profile: Union[AdminV1, UserV1] = Field(discriminator="role")
```

### Discriminator with Aliases

Handle discriminator fields with aliases:

```python
from pydantic import Field


@manager.model("Premium", "1.0.0")
class PremiumV1(BaseModel):
    account_type: Literal["premium"] = Field(
        default="premium",
        alias="accountType"
    )
    features: list[str]


@manager.model("Basic", "1.0.0")
class BasicV1(BaseModel):
    account_type: Literal["basic"] = Field(
        default="basic",
        alias="accountType"
    )
    ads_enabled: bool


@manager.model("Subscription", "1.0.0")
class SubscriptionV1(BaseModel):
    user_id: str
    tier: Union[PremiumV1, BasicV1] = Field(
        discriminator="account_type"  # Use field name, not alias
    )


# Data can use either field name or alias
subscription = SubscriptionV1(
    user_id="123",
    tier={"accountType": "premium", "features": ["ad-free", "hd"]}
)
```

## Adding New Union Members

Add new types to an existing union:

```python
# v1.0.0 - Two payment methods
@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV1, PayPalV1] = Field(discriminator="type")


# v2.0.0 - Add cryptocurrency payment
@manager.model("Crypto", "1.0.0")
class CryptoV1(BaseModel):
    type: Literal["crypto"] = "crypto"
    wallet_address: str
    currency: str


@manager.model("Order", "2.0.0", backward_compatible=True)
class OrderV2(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV2, PayPalV2, CryptoV1] = Field(
        discriminator="type"
    )


# No migration needed - old orders still work
# New orders can use crypto
@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    # Old payment methods migrate automatically
    return data


# Old order (credit card or PayPal)
old_order = {"order_id": "ORD-1", "payment_method": {"type": "paypal", "email": "..."}}
new_order = manager.migrate(old_order, "Order", "1.0.0", "2.0.0")
# Works - PayPal migrates to v2

# New order with crypto
crypto_order = OrderV2(
    order_id="ORD-2",
    payment_method={"type": "crypto", "wallet_address": "0x...", "currency": "BTC"}
)
```

## Removing Union Members

Handle deprecation of union members:

```python
# v1.0.0 - Three payment methods
@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV1, PayPalV1, CheckV1] = Field(
        discriminator="type"
    )


# v2.0.0 - Remove check payment (deprecated)
@manager.model("Order", "2.0.0")
class OrderV2(BaseModel):
    order_id: str
    payment_method: Union[CreditCardV2, PayPalV2] = Field(discriminator="type")


@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    """Convert check payments to bank transfers."""
    payment = data["payment_method"]

    # Handle removed payment type
    if payment.get("type") == "check":
        # Convert to bank transfer
        data["payment_method"] = {
            "type": "bank_transfer",
            "account_number": payment.get("account", "UNKNOWN"),
            "routing_number": payment.get("routing", "UNKNOWN")
        }

    return data
```

## Nested Discriminated Unions

Unions can contain other unions:

```python
# Personal payment methods
@manager.model("CreditCard", "1.0.0")
class CreditCardV1(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str


@manager.model("DebitCard", "1.0.0")
class DebitCardV1(BaseModel):
    type: Literal["debit_card"] = "debit_card"
    card_number: str


PersonalPayment = Union[CreditCardV1, DebitCardV1]


# Business payment methods
@manager.model("Invoice", "1.0.0")
class InvoiceV1(BaseModel):
    type: Literal["invoice"] = "invoice"
    invoice_number: str


@manager.model("PurchaseOrder", "1.0.0")
class PurchaseOrderV1(BaseModel):
    type: Literal["purchase_order"] = "purchase_order"
    po_number: str


BusinessPayment = Union[InvoiceV1, PurchaseOrderV1]


# Combined union
@manager.model("Transaction", "1.0.0")
class TransactionV1(BaseModel):
    transaction_id: str
    payment: Union[PersonalPayment, BusinessPayment] = Field(discriminator="type")
    # Flattens to: Union[CreditCardV1, DebitCardV1, InvoiceV1, PurchaseOrderV1]
```

## Lists of Discriminated Unions

Handle lists where each item can be different types:

```python
@manager.model("TextBlock", "1.0.0")
class TextBlockV1(BaseModel):
    type: Literal["text"] = "text"
    content: str


@manager.model("ImageBlock", "1.0.0")
class ImageBlockV1(BaseModel):
    type: Literal["image"] = "image"
    url: str
    alt_text: str


@manager.model("CodeBlock", "1.0.0")
class CodeBlockV1(BaseModel):
    type: Literal["code"] = "code"
    language: str
    code: str


ContentBlock = Union[TextBlockV1, ImageBlockV1, CodeBlockV1]


@manager.model("Document", "1.0.0")
class DocumentV1(BaseModel):
    title: str
    blocks: list[ContentBlock] = Field(discriminator="type")


# Document with mixed content types
doc = DocumentV1(
    title="Tutorial",
    blocks=[
        {"type": "text", "content": "Introduction"},
        {"type": "image", "url": "example.png", "alt_text": "Diagram"},
        {"type": "code", "language": "python", "code": "print('hello')"},
        {"type": "text", "content": "Conclusion"},
    ]
)


# Evolve content blocks
@manager.model("TextBlock", "2.0.0")
class TextBlockV2(BaseModel):
    type: Literal["text"] = "text"
    content: str
    format: str = "plain"  # New field


@manager.model("ImageBlock", "2.0.0")
class ImageBlockV2(BaseModel):
    type: Literal["image"] = "image"
    url: str
    alt_text: str
    width: int = 800  # New field
    height: int = 600  # New field


@manager.model("CodeBlock", "2.0.0")
class CodeBlockV2(BaseModel):
    type: Literal["code"] = "code"
    language: str
    code: str
    line_numbers: bool = True  # New field


ContentBlockV2 = Union[TextBlockV2, ImageBlockV2, CodeBlockV2]


@manager.model("Document", "2.0.0")
class DocumentV2(BaseModel):
    title: str
    blocks: list[ContentBlockV2] = Field(discriminator="type")


# Migrations for each block type
@manager.migration("TextBlock", "1.0.0", "2.0.0")
def migrate_text_block(data: ModelData) -> ModelData:
    return {**data, "format": "plain"}


@manager.migration("ImageBlock", "1.0.0", "2.0.0")
def migrate_image_block(data: ModelData) -> ModelData:
    return {**data, "width": 800, "height": 600}


@manager.migration("CodeBlock", "1.0.0", "2.0.0")
def migrate_code_block(data -> ModelData) -> ModelData:
    return {**data, "line_numbers": True}


@manager.migration("Document", "1.0.0", "2.0.0")
def migrate_document(data: ModelData) -> ModelData:
    # Each block in list migrates based on its type
    return data


# Migrate document - all blocks upgrade automatically
old_doc = {
    "title": "Tutorial",
    "blocks": [
        {"type": "text", "content": "Intro"},
        {"type": "image", "url": "pic.png", "alt_text": "Photo"},
        {"type": "code", "language": "python", "code": "print('hi')"},
    ]
}

new_doc = manager.migrate(old_doc, "Document", "1.0.0", "2.0.0")
# All blocks migrated according to their types
```

## Complex Union Hierarchies

Build sophisticated type hierarchies:

```python
# Base notification types
@manager.model("EmailNotification", "1.0.0")
class EmailNotificationV1(BaseModel):
    type: Literal["email"] = "email"
    to: str
    subject: str
    body: str


@manager.model("SMSNotification", "1.0.0")
class SMSNotificationV1(BaseModel):
    type: Literal["sms"] = "sms"
    phone: str
    message: str


@manager.model("PushNotification", "1.0.0")
class PushNotificationV1(BaseModel):
    type: Literal["push"] = "push"
    device_id: str
    title: str
    body: str


Notification = Union[EmailNotificationV1, SMSNotificationV1, PushNotificationV1]


# Event with notification
@manager.model("UserEvent", "1.0.0")
class UserEventV1(BaseModel):
    event_id: str
    user_id: str
    notification: Notification = Field(discriminator="type")


# System event (no notification)
@manager.model("SystemEvent", "1.0.0")
class SystemEventV1(BaseModel):
    event_id: str
    component: str
    severity: str


# All events
Event = Union[UserEventV1, SystemEventV1]


@manager.model("EventLog", "1.0.0")
class EventLogV1(BaseModel):
    timestamp: str
    events: list[Event] = Field(discriminator="event_id")  # Can't discriminate here!
```

**Note:** You can only discriminate on fields that exist in all union members.
For events with different structures, consider adding a common discriminator
field.

## Migration Strategies

### Type Conversion

Convert one union member type to another:

```python
@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    """Convert debit cards to credit cards (business decision)."""
    payment = data["payment_method"]

    if payment.get("type") == "debit_card":
        # Convert to credit card
        data["payment_method"] = {
            "type": "credit_card",
            "card_number": payment["card_number"],
            "expiry": payment.get("expiry", "12/99"),
            "cvv": "000"
        }

    return data
```

### Conditional Migration

Apply different logic based on union member:

```python
@manager.migration("Order", "1.0.0", "2.0.0")
def migrate_order(data: ModelData) -> ModelData:
    """Different handling based on payment type."""
    payment = data["payment_method"]
    payment_type = payment.get("type")

    if payment_type == "credit_card":
        # Credit cards: add fraud check flag
        payment["fraud_checked"] = True
    elif payment_type == "paypal":
        # PayPal: add transaction ID
        payment["transaction_id"] = f"PP-{data['order_id']}"
    elif payment_type == "bank_transfer":
        # Bank transfers: add processing status
        payment["status"] = "pending"

    return data
```

### Preserving Unknown Types

Handle forward compatibility:

```python
@manager.migration("Order", "2.0.0", "3.0.0")
def migrate_order(data: ModelData) -> ModelData:
    """Preserve payment methods we don't recognize."""
    payment = data["payment_method"]
    known_types = {"credit_card", "paypal", "bank_transfer", "crypto"}

    if payment.get("type") not in known_types:
        # Unknown type - preserve as-is and flag for review
        payment["_needs_review"] = True
        payment["_migrated_from"] = "2.0.0"

    return data
```

## Testing Discriminated Unions

Test each union member thoroughly:

```python
def test_payment_method_migrations() -> None:
    """Test all payment method types migrate correctly."""

    # Test credit card
    credit_card_results = manager.test_migration(
        "Order",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {
                    "order_id": "ORD-1",
                    "payment_method": {
                        "type": "credit_card",
                        "card_number": "4111",
                        "expiry": "12/25"
                    }
                },
                {
                    "order_id": "ORD-1",
                    "payment_method": {
                        "type": "credit_card",
                        "card_number": "4111",
                        "expiry": "12/25",
                        "cvv": "000"
                    }
                }
            )
        ]
    )
    credit_card_results.assert_all_passed()

    # Test PayPal
    paypal_results = manager.test_migration(
        "Order",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {
                    "order_id": "ORD-2",
                    "payment_method": {
                        "type": "paypal",
                        "email": "user@example.com"
                    }
                },
                {
                    "order_id": "ORD-2",
                    "payment_method": {
                        "type": "paypal",
                        "email": "user@example.com",
                        "verified": False
                    }
                }
            )
        ]
    )
    paypal_results.assert_all_passed()

    # Test bank transfer
    bank_results = manager.test_migration(
        "Order",
        "1.0.0",
        "2.0.0",
        test_cases=[
            (
                {
                    "order_id": "ORD-3",
                    "payment_method": {
                        "type": "bank_transfer",
                        "account_number": "123456",
                        "routing_number": "987654"
                    }
                },
                {
                    "order_id": "ORD-3",
                    "payment_method": {
                        "type": "bank_transfer",
                        "account_number": "123456",
                        "routing_number": "987654",
                        "swift_code": "UNKNOWN"
                    }
                }
            )
        ]
    )
    bank_results.assert_all_passed()
```

## Common Patterns

### Event Sourcing

Use discriminated unions for event streams:

```python
@manager.model("UserCreated", "1.0.0")
class UserCreatedV1(BaseModel):
    event_type: Literal["user_created"] = "user_created"
    user_id: str
    email: str


@manager.model("UserUpdated", "1.0.0")
class UserUpdatedV1(BaseModel):
    event_type: Literal["user_updated"] = "user_updated"
    user_id: str
    changes: dict[str, str]


@manager.model("UserDeleted", "1.0.0")
class UserDeletedV1(BaseModel):
    event_type: Literal["user_deleted"] = "user_deleted"
    user_id: str
    reason: str


UserEvent = Union[UserCreatedV1, UserUpdatedV1, UserDeletedV1]


@manager.model("EventStream", "1.0.0")
class EventStreamV1(BaseModel):
    timestamp: str
    event: UserEvent = Field(discriminator="event_type")
```

### API Response Envelopes

Handle different response types:

```python
@manager.model("SuccessResponse", "1.0.0")
class SuccessResponseV1(BaseModel):
    status: Literal["success"] = "success"
    data: dict[str, Any]


@manager.model("ErrorResponse", "1.0.0")
class ErrorResponseV1(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str


@manager.model("APIResponse", "1.0.0")
class APIResponseV1(BaseModel):
    request_id: str
    response: Union[SuccessResponseV1, ErrorResponseV1] = Field(
        discriminator="status"
    )
```

### Plugin Systems

Model different plugin types:

```python
@manager.model("DatabasePlugin", "1.0.0")
class DatabasePluginV1(BaseModel):
    plugin_type: Literal["database"] = "database"
    connection_string: str
    pool_size: int


@manager.model("CachePlugin", "1.0.0")
class CachePluginV1(BaseModel):
    plugin_type: Literal["cache"] = "cache"
    host: str
    ttl: int


@manager.model("LoggingPlugin", "1.0.0")
class LoggingPluginV1(BaseModel):
    plugin_type: Literal["logging"] = "logging"
    level: str
    output: str


Plugin = Union[DatabasePluginV1, CachePluginV1, LoggingPluginV1]


@manager.model("AppConfig", "1.0.0")
class AppConfigV1(BaseModel):
    app_name: str
    plugins: list[Plugin] = Field(discriminator="plugin_type")
```

## Best Practices

1. **Use descriptive discriminator values** - Match the model name or purpose
2. **Keep discriminator field consistent** - Usually `type` or `kind`
3. **Test all union members** - Each type needs migration tests
4. **Document union members** - List all possible types in docstrings
5. **Handle unknown types gracefully** - Forward compatibility matters
6. **Use Literal types** - Ensures type safety
7. **Consider versioning discriminators** - If discriminator values change

## Common Pitfalls

### Missing Discriminator Field

```python
# ❌ BAD - No discriminator field
@manager.model("Option1", "1.0.0")
class Option1V1(BaseModel):
    value: str


@manager.model("Option2", "1.0.0")
class Option2V1(BaseModel):
    value: int

# How does pyrmute know which is which?

# ✅ GOOD - Add discriminator
@manager.model("Option1", "1.0.0")
class Option1V1(BaseModel):
    type: Literal["option1"] = "option1"
    value: str
```

### Inconsistent Discriminator Values

```python
# ❌ BAD - Discriminator value changed
@manager.model("Payment", "1.0.0")
class PaymentV1(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str


@manager.model("Payment", "2.0.0")
class PaymentV2(BaseModel):
    type: Literal["card"] = "card"  # Changed!
    card_number: str

# Old data won't migrate correctly

# ✅ GOOD - Keep discriminator value stable
@manager.model("Payment", "2.0.0")
class PaymentV2(BaseModel):
    type: Literal["credit_card"] = "credit_card"  # Same
    card_number: str
```

### Forgetting to Register Union Members

```python
# ❌ BAD - Forgot to register one type
@manager.model("CreditCard", "1.0.0")
class CreditCardV1(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str


# PayPal not registered!
class PayPalV1(BaseModel):
    type: Literal["paypal"] = "paypal"
    email: str


@manager.model("Order", "1.0.0")
class OrderV1(BaseModel):
    payment: Union[CreditCardV1, PayPalV1] = Field(discriminator="type")


# ✅ GOOD - Register all union members
@manager.model("PayPal", "1.0.0")
class PayPalV1(BaseModel):
    type: Literal["paypal"] = "paypal"
    email: str
```

## Troubleshooting

### Union Member Not Migrating

Check that:

- All union member models are registered
- Migrations exist for each member type
- Discriminator field is correctly named
- Discriminator values match Literal values

### Wrong Migration Applied

Verify discriminator field spelling and casing:

```python
# Data uses "creditCard" but model expects "credit_card"
data = {"type": "creditCard", "card_number": "..."}  # Wrong!

# Fix: Match discriminator value exactly
data = {"type": "credit_card", "card_number": "..."}  # Correct
```

### Performance Issues with Large Lists

For lists with many discriminated unions, consider:

- Using batch processing
- Profiling to identify slow migrations
- Simplifying union member migrations

## Next Steps

Now that you understand discriminated union migrations:

**Continue learning:**

- [Nested Models](../user-guide/nested-models.md) - More on nested Pydantic
    models
- [Batch Processing](../user-guide/batch-processing.md) - Efficiently migrate
    lists of unions
- [Schema Generation](../user-guide/schema-generation.md) - Export schemas for
    discriminated unions

**Related topics:**

- [Writing Migrations](../user-guide/writing-migrations.md) - Best practices
    for union migrations
- [Testing Migrations](../user-guide/testing-migrations.md) - Test each union
    member type
