"""Message queue consumer with automatic schema migration.

Scenario: Multiple services publish events to a queue, but they're running different
versions and sending different schemas. Your consumer needs to handle all versions
gracefully.

- Messages arrive with `schema_version` metadata
- Consumer automatically migrates to current schema
- Processing logic only deals with current schema
- No need to coordinate deployments across all publishers

Example: Order processing system where:
- Old checkout service sends v1.0.0 orders
- Mobile app sends v2.0.0 orders
- New web app sends v3.0.0 orders
- Your fulfillment service handles them all
"""

import time
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel

from pyrmute import ModelData, ModelManager

manager = ModelManager()


class PaymentMethod(StrEnum):
    """Payment method types."""

    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"


class OrderStatus(StrEnum):
    """Order status."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


# v1.0.0 - Original checkout service (deprecated, but still in production)
@manager.model("OrderEvent", "1.0.0")
class OrderEventV1(BaseModel):
    """Original order schema from legacy checkout service."""

    order_id: str
    customer_email: str
    items: list[dict[str, Any]]  # Unstructured item data
    total: float  # Money as float (bad!)
    payment_method: str  # Free-form string
    timestamp: str  # Timestamp as string


# v2.0.0 - Mobile app (current production)
@manager.model("Address", "2.0.0", enable_ref=True)
class AddressV2(BaseModel):
    """Shipping address."""

    street: str
    city: str
    state: str
    postal_code: str
    country: str = "US"


@manager.model("OrderItem", "2.0.0", enable_ref=True)
class OrderItemV2(BaseModel):
    """Structured order item."""

    sku: str
    name: str
    quantity: int
    unit_price: Decimal


@manager.model("OrderEvent", "2.0.0")
class OrderEventV2(BaseModel):
    """Improved schema from mobile app."""

    order_id: str
    customer_email: str
    items: list[OrderItemV2]  # Structured items
    total_amount: Decimal  # Proper decimal
    payment_method: PaymentMethod  # Enum
    shipping_address: AddressV2
    timestamp: datetime  # Proper datetime


# v3.0.0 - New web app (latest)
@manager.model("Customer", "3.0.0", enable_ref=True)
class CustomerV3(BaseModel):
    """Full customer info."""

    customer_id: str
    email: str
    name: str
    phone: str | None = None


@manager.model("OrderEvent", "3.0.0")
class OrderEventV3(BaseModel):
    """Current schema with full features."""

    order_id: str
    customer: CustomerV3  # Full customer object
    items: list[OrderItemV2]
    subtotal: Decimal
    tax: Decimal
    shipping_cost: Decimal
    total_amount: Decimal
    payment_method: PaymentMethod
    shipping_address: AddressV2
    billing_address: AddressV2 | None = None
    status: OrderStatus = OrderStatus.PENDING
    notes: str | None = None
    timestamp: datetime
    source: str = "web"  # Track origin


@manager.migration("OrderEvent", "1.0.0", "2.0.0")
def migrate_order_v1_to_v2(data: ModelData) -> ModelData:
    """Migrate from legacy unstructured format to structured format."""
    items = [
        {
            "sku": item.get("id", "UNKNOWN"),
            "name": item.get("name", "Unknown Item"),
            "quantity": item.get("qty", 1),
            "unit_price": str(item.get("price", 0.0)),
        }
        for item in data.get("items", [])
    ]

    payment_str = data.get("payment_method", "").lower()
    if "paypal" in payment_str:
        payment = "paypal"
    elif "bank" in payment_str:
        payment = "bank_transfer"
    else:
        payment = "credit_card"

    return {
        "order_id": data["order_id"],
        "customer_email": data["customer_email"],
        "items": items,
        "total_amount": str(data.get("total", 0.0)),
        "payment_method": payment,
        "shipping_address": {
            "street": "Address not provided",
            "city": "Unknown",
            "state": "Unknown",
            "postal_code": "00000",
            "country": "US",
        },
        "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
    }


@manager.migration("OrderEvent", "2.0.0", "3.0.0")
def migrate_order_v2_to_v3(data: ModelData) -> ModelData:
    """Add customer object and cost breakdown."""
    email = data["customer_email"]
    customer_id = f"MIGRATED_{hash(email) % 100000}"

    total = Decimal(str(data["total_amount"]))
    tax = total * Decimal("0.08")  # Estimate 8% tax
    shipping = Decimal("5.99") if total < Decimal("50") else Decimal("0")
    subtotal = total - tax - shipping

    return {
        "order_id": data["order_id"],
        "customer": {
            "customer_id": customer_id,
            "email": email,
            "name": email.split("@")[0].title(),
            "phone": None,
        },
        "items": data["items"],
        "subtotal": str(subtotal),
        "tax": str(tax),
        "shipping_cost": str(shipping),
        "total_amount": str(total),
        "payment_method": data["payment_method"],
        "shipping_address": data["shipping_address"],
        "billing_address": None,
        "status": "pending",
        "notes": "Migrated from v2.0.0",
        "timestamp": data["timestamp"],
        "source": "mobile",
    }


class QueueMessage(BaseModel):
    """Message envelope with metadata."""

    schema_version: str
    payload: dict[str, Any]
    published_at: datetime
    publisher: str


# Simulated message queue
MOCK_QUEUE: list[QueueMessage] = [
    # Old messages from legacy system
    QueueMessage(
        schema_version="1.0.0",
        publisher="checkout-service-old",
        published_at=datetime(2023, 6, 15, 10, 30),
        payload={
            "order_id": "ORD-2023-001",
            "customer_email": "alice@example.com",
            "items": [
                {"id": "WIDGET-123", "name": "Blue Widget", "qty": 2, "price": 19.99},
                {"id": "GADGET-456", "name": "Red Gadget", "qty": 1, "price": 49.99},
            ],
            "total": 89.97,
            "payment_method": "Credit Card - Visa",
            "timestamp": "2023-06-15T10:30:00",
        },
    ),
    # Messages from mobile app
    QueueMessage(
        schema_version="2.0.0",
        publisher="mobile-app",
        published_at=datetime(2024, 3, 20, 14, 15),
        payload={
            "order_id": "ORD-2024-042",
            "customer_email": "bob@example.com",
            "items": [
                {
                    "sku": "LAPTOP-789",
                    "name": "Pro Laptop",
                    "quantity": 1,
                    "unit_price": "1299.99",
                }
            ],
            "total_amount": "1299.99",
            "payment_method": "paypal",
            "shipping_address": {
                "street": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94102",
                "country": "US",
            },
            "timestamp": "2024-03-20T14:15:00",
        },
    ),
    # Latest messages from new web app
    QueueMessage(
        schema_version="3.0.0",
        publisher="web-app-v3",
        published_at=datetime(2024, 10, 10, 9, 0),
        payload={
            "order_id": "ORD-2024-156",
            "customer": {
                "customer_id": "CUST-7890",
                "email": "charlie@example.com",
                "name": "Charlie Wilson",
                "phone": "+1-555-012-3456",
            },
            "items": [
                {
                    "sku": "PHONE-999",
                    "name": "Smartphone X",
                    "quantity": 1,
                    "unit_price": "899.00",
                },
                {
                    "sku": "CASE-111",
                    "name": "Phone Case",
                    "quantity": 2,
                    "unit_price": "15.99",
                },
            ],
            "subtotal": "930.98",
            "tax": "74.48",
            "shipping_cost": "0.00",
            "total_amount": "1005.46",
            "payment_method": "credit_card",
            "shipping_address": {
                "street": "456 Oak Ave",
                "city": "Portland",
                "state": "OR",
                "postal_code": "97201",
                "country": "US",
            },
            "billing_address": {
                "street": "789 Pine Rd",
                "city": "Seattle",
                "state": "WA",
                "postal_code": "98101",
                "country": "US",
            },
            "status": "confirmed",
            "notes": "Gift wrap requested",
            "timestamp": "2024-10-10T09:00:00",
            "source": "web",
        },
    ),
]


class OrderProcessor:
    """Order processing service that handles all schema versions."""

    def __init__(self) -> None:
        """Initializes the order processor."""
        self.processed_count = 0
        self.error_count = 0
        self.version_stats: dict[str, int] = {}

    def process_message(self, message: QueueMessage) -> None:
        """Process a single message, migrating to current schema."""
        print(f"\n{'=' * 80}")
        print(f"Processing message from {message.publisher}")
        print(f"Schema version: {message.schema_version}")
        print(f"Published at: {message.published_at}")
        print(f"{'=' * 80}")

        try:
            # Migrate to current schema
            order = cast(
                "OrderEventV3",
                manager.migrate(
                    message.payload,
                    "OrderEvent",
                    from_version=message.schema_version,
                    to_version="3.0.0",
                ),
            )

            # Now all business logic works with consistent schema
            self._fulfill_order(order)

            self.processed_count += 1
            self.version_stats[message.schema_version] = (
                self.version_stats.get(message.schema_version, 0) + 1
            )

            print(f"✓ Successfully processed order {order.order_id}")

        except Exception as e:
            print(f"✗ Error processing message: {e}")
            self.error_count += 1

    def _fulfill_order(self, order: OrderEventV3) -> None:
        """Business logic - always works with current schema."""
        print("\nFulfilling order:")
        print(f"  Order ID: {order.order_id}")
        print(f"  Customer: {order.customer.name} ({order.customer.email})")
        print(f"  Items: {len(order.items)}")
        for item in order.items:
            print(f"    - {item.name} (SKU: {item.sku}) x{item.quantity}")
        print(f"  Total: ${order.total_amount}")
        print(f"  Payment: {order.payment_method.value}")
        print(
            f"  Ship to: {order.shipping_address.city}, {order.shipping_address.state}"
        )
        print(f"  Status: {order.status.value}")
        if order.notes:
            print(f"  Notes: {order.notes}")

        # Business logic here...
        # - Validate inventory
        # - Process payment
        # - Create shipment
        # - Send confirmation email
        # etc.

    def print_stats(self) -> None:
        """Print processing statistics."""
        print(f"\n{'=' * 80}")
        print("PROCESSING STATISTICS")
        print(f"{'=' * 80}")
        print(f"Total processed: {self.processed_count}")
        print(f"Errors: {self.error_count}")
        print("\nBy schema version:")
        for version, count in sorted(self.version_stats.items()):
            print(f"  v{version}: {count} messages")
        print(f"{'=' * 80}")


def consume_queue() -> None:
    """Simulate consuming messages from a queue."""
    processor = OrderProcessor()

    print("Starting order processor...")
    print("Listening for messages from multiple publishers...\n")

    for message in MOCK_QUEUE:
        processor.process_message(message)
        time.sleep(0.25)  # Processing...

    processor.print_stats()


def example_rabbitmq_consumer() -> None:
    """Example: RabbitMQ consumer with schema migration.

    Install: pip install pika
    """
    print("\n# Example: RabbitMQ Consumer")
    print("""
import pika


def callback(ch, method, properties, body):
    message = json.loads(body)

    # Extract schema version from message headers
    schema_version = properties.headers.get("schema_version", "1.0.0")

    # Migrate to current schema
    order = cast(
        "OrderEventV3",
        manager.migrate(
            message, "OrderEvent", from_version=schema_version, to_version="3.0.0"
        ),
    )

    # Process with current schema
    process_order(order)

    ch.basic_ack(delivery_tag=method.delivery_tag)


connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.basic_consume(queue="orders", on_message_callback=callback)
channel.start_consuming()
    """)


def example_kafka_consumer() -> None:
    """Example: Kafka consumer with schema migration.

    Install: pip install kafka-python
    """
    print("\n# Example: Kafka Consumer")
    print("""
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "orders",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

for message in consumer:
    payload = message.value

    # Schema version in message metadata
    schema_version = payload.get("_schema_version", "1.0.0")

    # Migrate to current
    order = cast(
        "OrderEventV3",
        manager.migrate(
            payload, "OrderEvent", from_version=schema_version, to_version="3.0.0"
        ),
    )

    process_order(order)
    """)


def example_sqs_consumer() -> None:
    """Example: AWS SQS consumer with schema migration.

    Install: pip install boto3
    """
    print("\n# Example: AWS SQS Consumer")
    print("""
import boto3

sqs = boto3.client("sqs")
queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/orders"

while True:
    response = sqs.receive_message(
        QueueUrl=queue_url, MessageAttributes=["schema_version"]
    )

    for message in response.get("Messages", []):
        schema_version = message["MessageAttributes"]["schema_version"]["StringValue"]
        payload = json.loads(message["Body"])

        # Migrate and process
        order = cast(
            "OrderEventV3",
            manager.migrate(
                payload, "OrderEvent", from_version=schema_version, to_version="3.0.0"
            ),
        )

        process_order(order)

        # Delete message
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
    """)


if __name__ == "__main__":
    print("=" * 80)
    print("MESSAGE QUEUE CONSUMER - Schema Migration Example")
    print("=" * 80)
    print("\nScenario: Multiple services publish orders at different schema versions")
    print("Consumer migrates everything to current schema transparently")
    print()

    consume_queue()

    print("\n\n" + "=" * 80)
    print("REAL-WORLD INTEGRATIONS")
    print("=" * 80)
    example_rabbitmq_consumer()
    example_kafka_consumer()
    example_sqs_consumer()

    print("\n" + "=" * 80)
    print("This pattern works great for:")
    print("  ✓ Message queues (RabbitMQ, Kafka, SQS, etc.)")
    print("  ✓ Event streams")
    print("  ✓ Webhook consumers")
    print("  ✓ Any system receiving data from multiple sources")
    print("=" * 80)
