"""ETL/Data Import pipeline with automatic schema migration.

Scenario: You need to import historical data files (CSV, JSON, Excel) that were exported
over several years with evolving schemas. Each file has a different structure, and you
need to normalize everything into your current schema.

- Files have version identifiers (filename, metadata field, etc.)
- Automatically migrate to current schema
- Single processing pipeline handles all versions
- Easy to add new versions without changing import logic

Example: Customer data import from:
- 2022 exports: Basic contact info (v1.0.0)
- 2023 exports: Added company data (v2.0.0)
- 2024 exports: Full CRM with preferences (v3.0.0)
"""

import csv
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from pyrmute import ModelData, ModelManager

manager = ModelManager()


# v1.0.0 - 2022 exports (basic contact info)
@manager.model("Customer", "1.0.0")
class CustomerV1(BaseModel):
    """Original customer export format (2022)."""

    id: int
    name: str
    email: str
    phone: str | None = None
    created: str  # Date as string


# v2.0.0 - 2023 exports (added company data)
@manager.model("Customer", "2.0.0")
class CustomerV2(BaseModel):
    """2023 format with company info."""

    customer_id: int  # Renamed field
    full_name: str  # Renamed field
    email: str
    phone: str | None = None
    company_name: str | None = None
    job_title: str | None = None
    created_date: datetime  # Proper datetime
    last_contact: datetime | None = None


# v3.0.0 - Current format (full CRM with preferences)
@manager.model("Address", "3.0.0", enable_ref=True)
class AddressV3(BaseModel):
    """Customer address."""

    street: str
    city: str
    state: str
    postal_code: str
    country: str = "US"


@manager.model("Preferences", "3.0.0", enable_ref=True)
class PreferencesV3(BaseModel):
    """Customer preferences."""

    email_notifications: bool = True
    sms_notifications: bool = False
    newsletter: bool = True
    preferred_contact_time: str = "business_hours"


@manager.model("Customer", "3.0.0")
class CustomerV3(BaseModel):
    """Current customer schema with full CRM features."""

    customer_id: int
    first_name: str
    last_name: str
    email: str
    phone: str | None = None
    company_name: str | None = None
    job_title: str | None = None
    address: AddressV3 | None = None
    preferences: PreferencesV3
    lifetime_value: Decimal = Decimal("0.00")
    created_date: datetime
    last_contact: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


@manager.migration("Customer", "1.0.0", "2.0.0")
def migrate_customer_v1_to_v2(data: ModelData) -> ModelData:
    """Migrate 2022 format to 2023 format."""
    try:
        created = datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
    except Exception:
        created = datetime.now()

    return {
        "customer_id": data["id"],
        "full_name": data["name"],
        "email": data["email"],
        "phone": data.get("phone"),
        "company_name": None,
        "job_title": None,
        "created_date": created.isoformat(),
        "last_contact": None,
    }


@manager.migration("Customer", "2.0.0", "3.0.0")
def migrate_customer_v2_to_v3(data: ModelData) -> ModelData:
    """Migrate 2023 format to current format."""
    # Split full name into first/last
    full_name = data["full_name"]
    parts = full_name.split(maxsplit=1)
    first_name = parts[0] if parts else ""
    last_name = parts[1] if len(parts) > 1 else ""

    return {
        "customer_id": data["customer_id"],
        "first_name": first_name,
        "last_name": last_name,
        "email": data["email"],
        "phone": data.get("phone"),
        "company_name": data.get("company_name"),
        "job_title": data.get("job_title"),
        "address": None,
        "preferences": {
            "email_notifications": True,
            "sms_notifications": False,
            "newsletter": True,
            "preferred_contact_time": "business_hours",
        },
        "lifetime_value": "0.00",
        "created_date": data["created_date"],
        "last_contact": data.get("last_contact"),
        "tags": [],
        "notes": "Migrated from v2.0.0 export",
    }


class ImportStats:
    """Track import statistics."""

    def __init__(self) -> None:
        """Initializes the class."""
        self.total = 0
        self.success = 0
        self.errors = 0
        self.by_version: dict[str, int] = {}
        self.error_details: list[dict[str, Any]] = []

    def record_success(self, version: str) -> None:
        """Add success stats."""
        self.total += 1
        self.success += 1
        self.by_version[version] = self.by_version.get(version, 0) + 1

    def record_error(self, version: str, error: str, data: dict[str, Any]) -> None:
        """Add error stats."""
        self.total += 1
        self.errors += 1
        self.error_details.append(
            {
                "version": version,
                "error": error,
                "data": data,
            }
        )

    def print_summary(self) -> None:
        """Prints a summary to stdout."""
        print(f"\n{'=' * 80}")
        print("IMPORT SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total records: {self.total}")
        print(f"Successful: {self.success}")
        print(f"Errors: {self.errors}")
        print("\nBy schema version:")
        for version in sorted(self.by_version.keys()):
            count = self.by_version[version]
            print(f"  v{version}: {count} records")

        if self.error_details:
            print("\nFirst 5 errors:")
            for i, error in enumerate(self.error_details[:5], 1):
                print(f"\n  {i}. Version {error['version']}")
                print(f"     Error: {error['error']}")
                print(f"     Data: {error['data']}")
        print(f"{'=' * 80}")


def detect_csv_version(filepath: Path) -> str:
    """Detect schema version from CSV filename or content."""
    filename = filepath.name.lower()

    if "2022" in filename or "v1" in filename:
        return "1.0.0"
    if "2023" in filename or "v2" in filename:
        return "2.0.0"
    if "2024" in filename or "v3" in filename:
        return "3.0.0"

    # Default to oldest version for unknown files
    return "1.0.0"


def import_csv_file(filepath: Path, stats: ImportStats) -> list[CustomerV3]:
    """Import customers from a CSV file with automatic migration."""
    version = detect_csv_version(filepath)
    customers = []

    print(f"\nImporting {filepath.name}")
    print(f"  Detected schema version: {version}")

    with open(filepath) as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            try:
                # Migrate to current schema
                customer = cast(
                    "CustomerV3",
                    manager.migrate(
                        row, "Customer", from_version=version, to_version="3.0.0"
                    ),
                )
                customers.append(customer)
                stats.record_success(version)

            except Exception as e:
                stats.record_error(version, str(e), row)
                print(f"    ✗ Error on row {row_num}: {e}")

    print(f"  ✓ Imported {len(customers)} customers")
    return customers


def import_json_file(filepath: Path, stats: ImportStats) -> list[CustomerV3]:
    """Import customers from a JSON file with version metadata."""
    customers = []

    print(f"\nImporting {filepath.name}")

    with open(filepath) as f:
        data = json.load(f)

    # JSON files have version in metadata
    version = data.get("export_version", "1.0.0")
    records = data.get("customers", [])

    print(f"  Schema version: {version}")
    print(f"  Records: {len(records)}")

    for i, record in enumerate(records, start=1):
        try:
            customer = cast(
                "CustomerV3",
                manager.migrate(
                    record, "Customer", from_version=version, to_version="3.0.0"
                ),
            )
            customers.append(customer)
            stats.record_success(version)

        except Exception as e:
            stats.record_error(version, str(e), record)
            print(f"    ✗ Error on record {i}: {e}")

    print(f"  ✓ Imported {len(customers)} customers")
    return customers


def validate_and_deduplicate(customers: list[CustomerV3]) -> list[CustomerV3]:
    """Validate and remove duplicates."""
    print(f"\nValidating {len(customers)} customers...")

    # Deduplicate by email
    seen_emails = set()
    unique_customers = []

    for customer in customers:
        if customer.email not in seen_emails:
            seen_emails.add(customer.email)
            unique_customers.append(customer)

    duplicates = len(customers) - len(unique_customers)
    if duplicates > 0:
        print(f"  Removed {duplicates} duplicate(s)")

    print(f"  ✓ {len(unique_customers)} unique customers")
    return unique_customers


def save_to_database(customers: list[CustomerV3]) -> None:
    """Save customers to database (simulated)."""
    print(f"\nSaving {len(customers)} customers to database...")

    # for customer in customers:
    #     db.session.add(CustomerModel(**customer.model_dump()))
    # db.session.commit()

    print("  ✓ All customers saved")


def create_sample_files(data_dir: Path) -> None:
    """Create sample export files for testing."""
    data_dir.mkdir(exist_ok=True)

    # 2022 export (v1.0.0)
    csv_2022 = data_dir / "customers_2022_export.csv"
    with open(csv_2022, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "name", "email", "phone", "created"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "1001",
                    "name": "Alice Johnson",
                    "email": "alice@example.com",
                    "phone": "555-0101",
                    "created": "2022-01-15T10:30:00Z",
                },
                {
                    "id": "1002",
                    "name": "Bob Smith",
                    "email": "bob@example.com",
                    "phone": "",
                    "created": "2022-03-20T14:15:00Z",
                },
            ]
        )

    # 2023 export (v2.0.0)
    csv_2023 = data_dir / "customers_2023_export.csv"
    with open(csv_2023, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "customer_id",
                "full_name",
                "email",
                "phone",
                "company_name",
                "job_title",
                "created_date",
                "last_contact",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "customer_id": "2001",
                    "full_name": "Charlie Brown",
                    "email": "charlie@acme.com",
                    "phone": "555-0201",
                    "company_name": "Acme Corp",
                    "job_title": "CTO",
                    "created_date": "2023-05-10T09:00:00",
                    "last_contact": "2023-12-15T16:30:00",
                },
                {
                    "customer_id": "2002",
                    "full_name": "Diana Prince",
                    "email": "diana@example.com",
                    "phone": "555-0202",
                    "company_name": "Tech Startup",
                    "job_title": "CEO",
                    "created_date": "2023-08-22T11:45:00",
                    "last_contact": "",
                },
            ]
        )

    # 2024 export (v3.0.0) - JSON format
    json_2024 = data_dir / "customers_2024_export.json"
    export_data = {
        "export_version": "3.0.0",
        "export_date": "2024-10-10T00:00:00",
        "customers": [
            {
                "customer_id": 3001,
                "first_name": "Eve",
                "last_name": "Anderson",
                "email": "eve@example.com",
                "phone": "555-0301",
                "company_name": "Design Studio",
                "job_title": "Creative Director",
                "address": {
                    "street": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "postal_code": "94102",
                    "country": "US",
                },
                "preferences": {
                    "email_notifications": True,
                    "sms_notifications": True,
                    "newsletter": True,
                    "preferred_contact_time": "afternoon",
                },
                "lifetime_value": "15000.00",
                "created_date": "2024-01-05T10:00:00",
                "last_contact": "2024-10-01T14:30:00",
                "tags": ["vip", "enterprise"],
                "notes": "Key account - high priority",
            },
            {
                "customer_id": 3002,
                "first_name": "Frank",
                "last_name": "Miller",
                "email": "frank@example.com",
                "phone": None,
                "company_name": None,
                "job_title": None,
                "address": None,
                "preferences": {
                    "email_notifications": False,
                    "sms_notifications": False,
                    "newsletter": False,
                    "preferred_contact_time": "never",
                },
                "lifetime_value": "0.00",
                "created_date": "2024-09-15T08:00:00",
                "last_contact": None,
                "tags": ["inactive"],
                "notes": "Opted out of communications",
            },
        ],
    }

    with open(json_2024, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✓ Created sample files in {data_dir}/")


def run_import_pipeline(data_dir: Path) -> list[CustomerV3]:
    """Run the complete import pipeline."""
    print("=" * 80)
    print("ETL DATA IMPORT PIPELINE")
    print("=" * 80)
    print("\nImporting customer data from multiple years with different schemas")
    print("All data will be migrated to current schema (v3.0.0)\n")

    stats = ImportStats()
    all_customers = []

    csv_files = list(data_dir.glob("*.csv"))
    json_files = list(data_dir.glob("*.json"))

    print(f"Found {len(csv_files)} CSV file(s) and {len(json_files)} JSON file(s)")

    for filepath in sorted(csv_files):
        customers = import_csv_file(filepath, stats)
        all_customers.extend(customers)

    for filepath in sorted(json_files):
        customers = import_json_file(filepath, stats)
        all_customers.extend(customers)

    unique_customers = validate_and_deduplicate(all_customers)
    save_to_database(unique_customers)
    stats.print_summary()

    return unique_customers


def run_streaming_import(data_dir: Path) -> None:
    """Stream import large files without loading everything into memory."""
    print("\n" + "=" * 80)
    print("STREAMING IMPORT (for large files)")
    print("=" * 80)

    csv_file = data_dir / "customers_2022_export.csv"
    version = "1.0.0"
    batch_size = 1000

    print(f"\nStreaming import from {csv_file.name}")
    print(f"Batch size: {batch_size}")

    with open(csv_file) as f:
        reader = csv.DictReader(f)

        batch = []
        total_processed = 0

        for row in reader:
            batch.append(row)

            if len(batch) >= batch_size:
                migrated = manager.migrate_batch(
                    batch, "Customer", from_version=version, to_version="3.0.0"
                )

                # db.bulk_insert(migrated)

                total_processed += len(migrated)
                print(f"  Processed {total_processed} records...")

                batch = []

        if batch:
            migrated = manager.migrate_batch(
                batch, "Customer", from_version=version, to_version="3.0.0"
            )
            total_processed += len(migrated)

    print(f"✓ Streamed {total_processed} total records")


def test_migrations() -> None:
    """Test migration paths with sample data."""
    print("\n" + "=" * 80)
    print("TESTING MIGRATIONS")
    print("=" * 80)

    test_cases = [
        # v1 -> v3
        (
            {
                "id": 999,
                "name": "Test User",
                "email": "test@example.com",
                "phone": "555-9999",
                "created": "2022-01-01T00:00:00Z",
            },
            "1.0.0",
        ),
        # v2 -> v3
        (
            {
                "customer_id": 888,
                "full_name": "Another Test",
                "email": "another@example.com",
                "phone": "555-8888",
                "company_name": "Test Co",
                "job_title": "Tester",
                "created_date": "2023-01-01T00:00:00",
                "last_contact": None,
            },
            "2.0.0",
        ),
    ]

    for i, (data, from_version) in enumerate(test_cases, 1):
        print(f"\nTest {i}: Migrating from v{from_version} to v3.0.0")
        print(f"  Input: {data}")

        try:
            result = manager.migrate(data, "Customer", from_version, "3.0.0")
            print(f"  ✓ Output: {result}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Show schema diff
    print("\n" + "=" * 80)
    print("SCHEMA CHANGES")
    print("=" * 80)

    diff = manager.diff("Customer", "1.0.0", "3.0.0")
    print("\nChanges from v1.0.0 to v3.0.0:")
    print(diff.to_markdown())


def example_excel_import() -> None:
    """Example: Import from Excel files.

    Install: pip install openpyxl pandas
    """
    print("\n# Example: Excel Import")
    print("""
import pandas as pd


def import_excel(filepath: Path, version: str) -> list[CustomerV3]:
    df = pd.read_excel(filepath)
    records = df.to_dict("records")

    customers = manager.migrate_batch(
        records,
        "Customer",
        from_version=version,
        to_version="3.0.0",
        parallel=True,  # Use parallel processing for large files
        max_workers=4,
    )

    return customers
    """)


def example_api_export_import() -> None:
    """Example: Import from legacy API exports."""
    print("\n# Example: Legacy API Export Import")
    print("""
import requests


def import_from_legacy_api() -> None:
    response = requests.get("https://old-api.example.com/customers/export")
    data = response.json()

    # Old API uses v1.0.0 schema
    version = data.get("schema_version", "1.0.0")
    customers_data = data["customers"]

    # Stream migration for large datasets
    for customer in manager.migrate_batch_streaming(
        customers_data,
        "Customer",
        from_version=version,
        to_version="3.0.0",
        chunk_size=100,
    ):
        # Process and save as they're migrated
        db.session.add(customer)

    db.session.commit()
    """)


def example_database_export() -> None:
    """Example: Migrate data between databases."""
    print("\n# Example: Database-to-Database Migration")
    print("""
from sqlalchemy import create_engine

old_engine = create_engine("postgresql://old-db/customers")
new_engine = create_engine("postgresql://new-db/customers")

# Read from old database
old_data = pd.read_sql("SELECT * FROM customers", old_engine)

migrated = manager.migrate_batch_data(
    old_data.to_dict("records"),
    "Customer",
    from_version="1.0.0",
    to_version="3.0.0",
    parallel=True,
)

# Write to new database
new_df = pd.DataFrame(migrated)
new_df.to_sql("customers", new_engine, if_exists="append", index=False)
""")


if __name__ == "__main__":
    # Create sample data files
    data_dir = Path("./sample_exports")
    create_sample_files(data_dir)

    # Run the import pipeline
    customers = run_import_pipeline(data_dir)

    # Show some imported customers
    print("\n" + "=" * 80)
    print("SAMPLE IMPORTED CUSTOMERS")
    print("=" * 80)
    for i, customer in enumerate(customers[:3], 1):
        print(f"\nCustomer {i}:")
        print(f"  ID: {customer.customer_id}")
        print(f"  Name: {customer.first_name} {customer.last_name}")
        print(f"  Email: {customer.email}")
        print(f"  Company: {customer.company_name or 'N/A'}")
        if customer.address:
            print(f"  Location: {customer.address.city}, {customer.address.state}")
        print(f"  LTV: ${customer.lifetime_value}")
        print(f"  Tags: {', '.join(customer.tags) if customer.tags else 'None'}")

    # Test migrations
    test_migrations()

    # Show streaming example (commented out to avoid recreating files)
    # run_streaming_import(data_dir)

    # Show additional examples
    print("\n\n" + "=" * 80)
    print("ADDITIONAL USE CASES")
    print("=" * 80)
    example_excel_import()
    example_api_export_import()
    example_database_export()

    print("\n" + "=" * 80)
    print("This pattern is perfect for:")
    print("  ✓ ETL pipelines")
    print("  ✓ Data imports from multiple sources")
    print("  ✓ Historical data migration")
    print("  ✓ Batch processing with schema evolution")
    print("  ✓ Database migration scripts")
    print("=" * 80)

    # Cleanup
    import shutil

    print("\nCleaning up sample files...")
    shutil.rmtree(data_dir)
    print("✓ Done!")
