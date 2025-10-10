"""Advanced features example."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from pyrmute import MigrationTestCase, ModelData, ModelManager

manager = ModelManager()


class UserRole(StrEnum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class SubscriptionTier(StrEnum):
    """Subscription tiers."""

    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@manager.model("Address", "1.0.0", enable_ref=True)
class AddressV1(BaseModel):
    """Address with validation."""

    street: str
    city: str
    country: str = "USA"
    postal_code: Annotated[str, Field(pattern=r"^\d{5}(-\d{4})?$")]


@manager.model("PaymentMethod", "1.0.0", enable_ref=True)
class PaymentMethodV1(BaseModel):
    """Payment method with discriminated union."""

    type: Literal["credit_card", "paypal", "bank"]
    account_id: str
    is_default: bool = False


@manager.model("User", "1.0.0")
class UserV1(BaseModel):
    """Initial user model with many Pydantic features."""

    username: Annotated[str, Field(min_length=3, max_length=50)]
    email: str
    role: UserRole = UserRole.USER

    address: AddressV1 | None = None
    payment_methods: list[PaymentMethodV1] = Field(default_factory=list)

    age: Annotated[int, Field(ge=18, le=120)] | None = None
    balance: Annotated[Decimal, Field(ge=0, decimal_places=2)] = Decimal("0.00")

    tags: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    metadata: dict[str, int | str | bool] = Field(default_factory=dict)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Incomplete validation."""
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()


@manager.model("Address", "2.0.0", enable_ref=True)
class AddressV2(BaseModel):
    """Address with split name fields and validation."""

    street_line1: str
    street_line2: str | None = None
    city: str
    state: str | None = None  # New field
    country: str = "USA"
    postal_code: str
    is_verified: bool = False  # New field


@manager.model("CreditCard", "2.0.0", enable_ref=True)
class CreditCardV2(BaseModel):
    """Specific credit card model."""

    type: Literal["CreditCard"] = "CreditCard"
    card_number: Annotated[str, Field(pattern=r"^\d{16}$")]
    expiry: str
    cvv: Annotated[str, Field(pattern=r"^\d{3,4}$")]
    is_default: bool = False


@manager.model("PayPal", "2.0.0", enable_ref=True)
class PayPalV2(BaseModel):
    """Specific PayPal model."""

    type: Literal["PayPal"] = "PayPal"
    email: str
    is_default: bool = False


@manager.model("BankAccount", "2.0.0", enable_ref=True)
class BankAccountV2(BaseModel):
    """Specific bank account model."""

    type: Literal["BankAccount"] = "BankAccount"
    account_number: str
    routing_number: str
    account_type: Literal["checking", "savings"]
    is_default: bool = False


@manager.model("Subscription", "2.0.0", enable_ref=True)
class SubscriptionV2(BaseModel):
    """New subscription model."""

    tier: SubscriptionTier
    start_date: datetime
    end_date: datetime | None = None
    auto_renew: bool = True


@manager.model("User", "2.0.0")
class UserV2(BaseModel):
    """Refactored user with discriminated unions and new features."""

    # Renamed field
    username: Annotated[str, Field(min_length=3, max_length=50)]
    primary_email: str  # Renamed from 'email'
    secondary_emails: list[str] = Field(default_factory=list)  # New field

    # Role split into role and permissions
    role: UserRole = UserRole.USER
    permissions: set[str] = Field(default_factory=set)  # New field

    # Nested models
    addresses: list[AddressV2] = Field(default_factory=list)  # Changed to list

    # Discriminated union for payment methods
    payment_methods: list[
        Annotated[
            CreditCardV2 | PayPalV2 | BankAccountV2,
            Field(discriminator="type"),
        ]
    ] = Field(default_factory=list)

    # Type changes
    age: int | None = None  # Removed constraints
    balance: Decimal = Decimal("0.00")  # Removed constraints
    credit_limit: Decimal | None = None  # New field

    # New subscription model
    subscription: SubscriptionV2 | None = None

    # Modified collections
    tags: set[str] = Field(default_factory=set)  # Changed from list to set
    metadata: dict[str, str] = Field(default_factory=dict)  # Restricted to str values

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_name(self) -> str:
        """New computed field."""
        return f"{self.username} ({self.role.value})"

    @field_validator("primary_email", "secondary_emails")
    @classmethod
    def validate_email(cls, v: str | list[str]) -> str | list[str]:
        """Can be a list."""
        if isinstance(v, str):
            if "@" not in v:
                raise ValueError("Invalid email")
            return v.lower()
        return [email.lower() for email in v if "@" in email]

    @model_validator(mode="after")
    def validate_credit_limit(self) -> "UserV2":
        """After validation."""
        if self.role == UserRole.GUEST and self.credit_limit:
            raise ValueError("Guests cannot have credit limits")
        return self


@manager.migration("Address", "1.0.0", "2.0.0")
def migrate_address(data: ModelData) -> ModelData:
    """Migrate address from v1 to v2."""
    return {
        "street_line1": data["street"],
        "street_line2": None,
        "city": data["city"],
        "state": None,
        "country": data.get("country", "USA"),
        "postal_code": data["postal_code"],
        "is_verified": False,
    }


def migrate_payment_method_v1_to_v2(data: ModelData) -> ModelData:
    """Helper to migrate payment method from v1 to v2 structure."""
    payment_type = data["type"]

    if payment_type == "credit_card":
        return {
            "type": "CreditCard",
            "card_number": data["account_id"].zfill(16),
            "expiry": "12/99",
            "cvv": "000",
            "is_default": data.get("is_default", False),
        }
    if payment_type == "paypal":
        return {
            "type": "PayPal",
            "email": f"{data['account_id']}@paypal.com",
            "is_default": data.get("is_default", False),
        }
    return {
        "type": "BankAccount",
        "account_number": data["account_id"],
        "routing_number": "000000000",
        "account_type": "checking",
        "is_default": data.get("is_default", False),
    }


@manager.migration("User", "1.0.0", "2.0.0")
def migrate_user(data: ModelData) -> ModelData:
    """Complex user migration."""
    migrated = {
        "username": data["username"],
        "primary_email": data["email"],
        "secondary_emails": [],
        "role": data.get("role", "user"),
        "permissions": set(),
        "addresses": [],
        "payment_methods": [],
        "age": data.get("age"),
        "balance": str(data.get("balance", "0.00")),
        "credit_limit": None,
        "subscription": None,
        "tags": list(set(data.get("tags", []))),
        "metadata": {k: str(v) for k, v in data.get("metadata", {}).items()},
    }

    if data.get("address"):
        address_data = manager.migrate_data(
            data["address"], "Address", "1.0.0", "2.0.0"
        )
        migrated["addresses"] = [address_data]

    for pm in data.get("payment_methods", []):
        pm_data = migrate_payment_method_v1_to_v2(pm)
        migrated["payment_methods"].append(pm_data)

    if migrated["role"] == "admin":
        migrated["permissions"] = {"read", "write", "delete", "manage_users"}
    elif migrated["role"] == "user":
        migrated["permissions"] = {"read", "write"}
    else:
        migrated["permissions"] = {"read"}

    return migrated


if __name__ == "__main__":
    user_v1_data = {
        "username": "johndoe",
        "email": "John@Example.COM",
        "role": "admin",
        "age": 30,
        "balance": "1234.56",
        "tags": ["python", "pydantic", "python"],
        "metadata": {"login_count": 42, "verified": True, "tier": "premium"},
        "address": {
            "street": "123 Main St",
            "city": "Springfield",
            "country": "USA",
            "postal_code": "12345",
        },
        "payment_methods": [
            {
                "type": "credit_card",
                "account_id": "1234567890123456",
                "is_default": True,
            },
            {"type": "paypal", "account_id": "john.doe"},
            {"type": "bank", "account_id": "9876543210", "is_default": False},
        ],
    }

    print("=" * 80)
    print("TESTING COMPLEX MIGRATION")
    print("=" * 80)

    user_v2 = cast("UserV2", manager.migrate(user_v1_data, "User", "1.0.0", "2.0.0"))
    print("\n✓ Migration successful!")
    print(f"\nMigrated user: {user_v2}")
    print(f"\nDisplay name: {user_v2.display_name}")
    print(f"Permissions: {user_v2.permissions}")
    print(f"Tags (now set): {user_v2.tags}")
    print(f"Payment methods: {len(user_v2.payment_methods)}")

    print("\n" + "=" * 80)
    print("MODEL DIFFERENCES")
    print("=" * 80)
    diff = manager.diff("User", "1.0.0", "2.0.0")
    print(diff.to_markdown())

    print("\n" + "=" * 80)
    print("BATCH MIGRATION TEST")
    print("=" * 80)

    batch_data = [
        {
            "username": f"user{i}",
            "email": f"user{i}@example.com",
            "role": "user",
            "tags": [f"tag{i}"],
        }
        for i in range(5)
    ]

    users_v2 = manager.migrate_batch(batch_data, "User", "1.0.0", "2.0.0")
    print(f"✓ Migrated {len(users_v2)} users in batch")

    print("\n" + "=" * 80)
    print("MIGRATION TESTING")
    print("=" * 80)

    test_cases: list[tuple[ModelData, ModelData] | MigrationTestCase] = [
        (
            {"username": "alice", "email": "alice@example.com", "role": "user"},
            {
                "username": "alice",
                "primary_email": "alice@example.com",
                "secondary_emails": [],
                "role": "user",
                "permissions": {"read", "write"},
                "addresses": [],
                "payment_methods": [],
                "age": None,
                "balance": "0.00",
                "credit_limit": None,
                "subscription": None,
                "tags": [],
                "metadata": {},
            },
        ),
    ]

    results = manager.test_migration("User", "1.0.0", "2.0.0", test_cases)
    print(f"✓ Test passed: {results.all_passed}")
    if not results.all_passed:
        for failure in results.failures:
            print(f"  Failed: {failure.error}")
            print(f"  Expected: {failure.test_case.target}")
            print(f"  Got: {failure.actual}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
