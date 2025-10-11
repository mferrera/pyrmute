"""ML Model Inference Pipeline with Schema Versioning.

Scenario: You have multiple model versions deployed simultaneously:
- Legacy models (v1, v2) still serving some users
- Current production model (v3)
- Experimental model (v4) in A/B testing

Each model version expects different input/output schemas as features evolve.
Your inference service needs to:
1. Accept requests in any schema version
2. Route to appropriate model
3. Normalize outputs to a consistent format
4. Log predictions for retraining

- Feature engineering evolves over time
- Different model versions need different inputs
- Prediction outputs have different structures
- Need to aggregate predictions for analysis

Example: Product recommendation system where:
- v1.0.0: Simple collaborative filtering (user_id, item_ids)
- v2.0.0: Added content features (descriptions, categories)
- v3.0.0: Deep learning with embeddings
- v4.0.0: Multi-modal with images and text
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from pyrmute import ModelData, ModelManager

manager = ModelManager()


class ModelType(str, Enum):
    """ML model architectures."""

    COLLABORATIVE_FILTER = "collaborative_filter"
    CONTENT_BASED = "content_based"
    DEEP_LEARNING = "deep_learning"
    MULTIMODAL = "multimodal"


# v1.0.0 - Simple collaborative filtering (2022)
@manager.model("RecommendationRequest", "1.0.0")
class RecommendationRequestV1(BaseModel):
    """Original model - basic collaborative filtering."""

    user_id: int
    num_recommendations: int = 5
    exclude_items: list[int] = Field(default_factory=list)


@manager.model("RecommendationResponse", "1.0.0")
class RecommendationResponseV1(BaseModel):
    """Simple recommendations with scores."""

    user_id: int
    recommended_items: list[int]
    scores: list[float]
    model_version: str = "1.0.0"


# v2.0.0 - Content-based with metadata (2023)
@manager.model("UserContext", "2.0.0", enable_ref=True)
class UserContextV2(BaseModel):
    """User context for better recommendations."""

    user_id: int
    recent_views: list[int] = Field(default_factory=list)
    recent_purchases: list[int] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)


@manager.model("RecommendationRequest", "2.0.0")
class RecommendationRequestV2(BaseModel):
    """Enhanced request with user context."""

    user_context: UserContextV2
    num_recommendations: int = 5
    exclude_items: list[int] = Field(default_factory=list)
    category_filter: list[str] | None = None


@manager.model("ItemMetadata", "2.0.0", enable_ref=True)
class ItemMetadataV2(BaseModel):
    """Item details in response."""

    item_id: int
    title: str
    category: str
    price: float


@manager.model("RecommendationResponse", "2.0.0")
class RecommendationResponseV2(BaseModel):
    """Recommendations with metadata."""

    user_id: int
    items: list[ItemMetadataV2]
    scores: list[float]
    reasoning: list[str]  # Why each item was recommended
    model_version: str = "2.0.0"


# v3.0.0 - Deep learning with embeddings (2024)
@manager.model("UserEmbedding", "3.0.0", enable_ref=True)
class UserEmbeddingV3(BaseModel):
    """User representation vector."""

    user_id: int
    embedding: list[float]  # 128-dim vector
    last_updated: datetime


@manager.model("RecommendationRequest", "3.0.0")
class RecommendationRequestV3(BaseModel):
    """Request with pre-computed embeddings."""

    user_embedding: UserEmbeddingV3 | None = None
    user_id: int | None = None  # Fallback if embedding not available
    num_recommendations: int = 5
    exclude_items: list[int] = Field(default_factory=list)
    diversity_weight: float = 0.5  # Balance relevance vs diversity
    exploration_rate: float = 0.1  # A/B testing parameter


@manager.model("ItemPrediction", "3.0.0", enable_ref=True)
class ItemPredictionV3(BaseModel):
    """Detailed prediction for each item."""

    item_id: int
    title: str
    category: str
    price: float
    predicted_rating: float
    confidence: float
    explanation_factors: dict[str, float]  # Feature contributions


@manager.model("RecommendationResponse", "3.0.0")
class RecommendationResponseV3(BaseModel):
    """Rich predictions with explainability."""

    user_id: int
    predictions: list[ItemPredictionV3]
    model_version: str = "3.0.0"
    model_type: ModelType = ModelType.DEEP_LEARNING
    inference_time_ms: float
    ab_test_variant: str | None = None


# v4.0.0 - Multi-modal (experimental)
@manager.model("MultiModalFeatures", "4.0.0", enable_ref=True)
class MultiModalFeaturesV4(BaseModel):
    """Combined features from multiple modalities."""

    user_id: int
    text_embedding: list[float]  # From user reviews/searches
    image_embedding: list[float] | None = None  # From user uploads
    behavioral_features: dict[str, float]  # Clickstream, dwell time, etc.
    temporal_features: dict[str, float]  # Time-of-day, seasonality


@manager.model("RecommendationRequest", "4.0.0")
class RecommendationRequestV4(BaseModel):
    """Multi-modal request."""

    features: MultiModalFeaturesV4
    num_recommendations: int = 5
    exclude_items: list[int] = Field(default_factory=list)
    personalization_strength: float = 1.0
    use_llm_reranking: bool = False  # Use LLM for final ranking


@manager.model("RecommendationResponse", "4.0.0")
class RecommendationResponseV4(BaseModel):
    """Multi-modal predictions with LLM explanations."""

    user_id: int
    predictions: list[ItemPredictionV3]
    llm_explanation: str | None = None  # Natural language explanation
    model_version: str = "4.0.0"
    model_type: ModelType = ModelType.MULTIMODAL
    inference_time_ms: float
    ab_test_variant: str
    model_ensemble: list[str]  # Which models contributed


@manager.migration("RecommendationRequest", "1.0.0", "2.0.0")
def migrate_request_v1_to_v2(data: ModelData) -> ModelData:
    """Add user context to v1 requests."""
    return {
        "user_context": {
            "user_id": data["user_id"],
            "recent_views": [],
            "recent_purchases": [],
            "preferred_categories": [],
        },
        "num_recommendations": data.get("num_recommendations", 5),
        "exclude_items": data.get("exclude_items", []),
        "category_filter": None,
    }


@manager.migration("RecommendationRequest", "2.0.0", "3.0.0")
def migrate_request_v2_to_v3(data: ModelData) -> ModelData:
    """Convert to embedding-based request."""
    return {
        "user_embedding": None,  # Will be computed on the fly
        "user_id": data["user_context"]["user_id"],
        "num_recommendations": data.get("num_recommendations", 5),
        "exclude_items": data.get("exclude_items", []),
        "diversity_weight": 0.5,
        "exploration_rate": 0.1,
    }


@manager.migration("RecommendationRequest", "3.0.0", "4.0.0")
def migrate_request_v3_to_v4(data: ModelData) -> ModelData:
    """Convert to multi-modal request."""
    user_id = data.get("user_id") or (
        data["user_embedding"]["user_id"] if data.get("user_embedding") else 0
    )

    text_embedding = (
        data["user_embedding"]["embedding"]
        if data.get("user_embedding")
        else [0.0] * 128
    )

    return {
        "features": {
            "user_id": user_id,
            "text_embedding": text_embedding,
            "image_embedding": None,
            "behavioral_features": {},
            "temporal_features": {},
        },
        "num_recommendations": data.get("num_recommendations", 5),
        "exclude_items": data.get("exclude_items", []),
        "personalization_strength": 1.0,
        "use_llm_reranking": False,
    }


@manager.migration("RecommendationResponse", "1.0.0", "3.0.0")
def migrate_response_v1_to_v3(data: ModelData) -> ModelData:
    """Normalize v1 responses to v3 format."""
    predictions = []

    for item_id, score in zip(
        data["recommended_items"],
        data.get("scores", [0.0] * len(data["recommended_items"])),
        strict=False,
    ):
        predictions.append(
            {
                "item_id": item_id,
                "title": f"Item {item_id}",
                "category": "Unknown",
                "price": 0.0,
                "predicted_rating": score,
                "confidence": 0.5,
                "explanation_factors": {"legacy_score": score},
            }
        )

    return {
        "user_id": data["user_id"],
        "predictions": predictions,
        "model_version": data.get("model_version", "1.0.0"),
        "model_type": "collaborative_filter",
        "inference_time_ms": 0.0,
        "ab_test_variant": None,
    }


@manager.migration("RecommendationResponse", "2.0.0", "3.0.0")
def migrate_response_v2_to_v3(data: ModelData) -> ModelData:
    """Normalize v2 responses to v3 format."""
    predictions = []

    for item, score, reason in zip(
        data["items"],
        data["scores"],
        data.get("reasoning", [""] * len(data["items"])),
        strict=False,
    ):
        predictions.append(
            {
                "item_id": item["item_id"],
                "title": item["title"],
                "category": item["category"],
                "price": item["price"],
                "predicted_rating": score,
                "confidence": 0.7,
                "explanation_factors": {
                    "content_similarity": score,
                    "reasoning": reason,
                },
            }
        )

    return {
        "user_id": data["user_id"],
        "predictions": predictions,
        "model_version": data.get("model_version", "2.0.0"),
        "model_type": "content_based",
        "inference_time_ms": 0.0,
        "ab_test_variant": None,
    }


@manager.migration("RecommendationResponse", "3.0.0", "4.0.0")
def migrate_response_v3_to_v4(data: ModelData) -> ModelData:
    """Add v4 features to v3 response."""
    return {
        **data,
        "llm_explanation": None,
        "model_version": "4.0.0",
        "model_type": data.get("model_type", "deep_learning"),
        "ab_test_variant": "control",
        "model_ensemble": [data.get("model_type", "deep_learning")],
    }


@manager.migration("RecommendationResponse", "4.0.0", "3.0.0")
def migrate_response_v4_to_v3(data: ModelData) -> ModelData:
    """Downgrade v4 response to v3 format (for logging/analytics)."""
    result = dict(data.items())
    # Remove v4-specific fields
    result.pop("llm_explanation", None)
    result.pop("model_ensemble", None)
    result["model_version"] = "3.0.0"
    return result


class ModelRegistry:
    """Registry of deployed models."""

    def __init__(self) -> None:
        """Sets the mapping of deployed models."""
        self.models = {
            "1.0.0": {"status": "deprecated", "load": 0.05},
            "2.0.0": {"status": "supported", "load": 0.15},
            "3.0.0": {"status": "production", "load": 0.70},
            "4.0.0": {"status": "experimental", "load": 0.10},
        }

    def get_model_for_request(self, request_version: str, user_id: int) -> str:
        """Route to appropriate model version."""
        # A/B testing: some users get experimental model
        if user_id % 10 == 0:
            return "4.0.0"
        # Legacy users stay on old models
        if request_version == "1.0.0":
            return "1.0.0"
        # Default to production
        return "3.0.0"


class InferenceService:
    """ML inference service with schema migration."""

    def __init__(self) -> None:
        """Initializes the inference service."""
        self.registry = ModelRegistry()
        self.prediction_log: list[ModelData] = []

    def predict(
        self, request_data: dict[str, Any], request_version: str
    ) -> dict[str, Any]:
        """Handle prediction request with automatic schema migration."""
        print(f"\n{'=' * 80}")
        print(f"Incoming request (schema v{request_version})")
        print(f"{'=' * 80}")

        user_id = self._extract_user_id(request_data, request_version)
        print(f"User ID: {user_id}")

        target_model_version = self.registry.get_model_for_request(
            request_version, user_id
        )
        print(f"Target model: v{target_model_version}")

        if request_version != target_model_version:
            print(f"Migrating request: v{request_version} → v{target_model_version}")
            request_data = manager.migrate_data(
                request_data,
                "RecommendationRequest",
                from_version=request_version,
                to_version=target_model_version,
            )

        start_time = datetime.now()
        response_data = self._run_model(request_data, target_model_version, user_id)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        response_data["inference_time_ms"] = inference_time

        normalized_response = manager.migrate_data(
            response_data,
            "RecommendationResponse",
            from_version=target_model_version,
            to_version="3.0.0",
        )

        # Log for retraining
        self._log_prediction(request_version, target_model_version, normalized_response)

        print(f"✓ Prediction complete ({inference_time:.2f}ms)")

        return response_data

    def _extract_user_id(self, request: dict[str, Any], version: str) -> int:
        """Extract user ID from different request formats."""
        if version == "1.0.0":
            return int(request["user_id"])
        if version == "2.0.0":
            return int(request["user_context"]["user_id"])
        if version == "3.0.0":
            return int(request.get("user_id") or request["user_embedding"]["user_id"])
        # 4.0.0
        return int(request["features"]["user_id"])

    def _run_model(
        self, request: dict[str, Any], model_version: str, user_id: int
    ) -> dict[str, Any]:
        """Simulate model inference (would call actual ML model)."""
        # In production, this would:
        # - Load model from model store
        # - Run inference
        # - Apply business rules
        # - Return predictions

        # Simulated predictions
        if model_version == "1.0.0":
            return {
                "user_id": user_id,
                "recommended_items": [101, 102, 103, 104, 105],
                "scores": [0.95, 0.87, 0.82, 0.79, 0.76],
                "model_version": "1.0.0",
            }
        if model_version == "2.0.0":
            return {
                "user_id": user_id,
                "items": [
                    {
                        "item_id": 201,
                        "title": "Premium Headphones",
                        "category": "Electronics",
                        "price": 299.99,
                    },
                    {
                        "item_id": 202,
                        "title": "Wireless Mouse",
                        "category": "Electronics",
                        "price": 49.99,
                    },
                    {
                        "item_id": 203,
                        "title": "USB-C Cable",
                        "category": "Accessories",
                        "price": 19.99,
                    },
                ],
                "scores": [0.92, 0.85, 0.78],
                "reasoning": [
                    "Based on recent electronics purchases",
                    "Complements recent purchases",
                    "Frequently bought together",
                ],
                "model_version": "2.0.0",
            }
        if model_version == "3.0.0":
            return {
                "user_id": user_id,
                "predictions": [
                    {
                        "item_id": 301,
                        "title": "4K Monitor",
                        "category": "Electronics",
                        "price": 599.99,
                        "predicted_rating": 4.7,
                        "confidence": 0.89,
                        "explanation_factors": {
                            "embedding_similarity": 0.85,
                            "category_preference": 0.92,
                            "price_range_fit": 0.88,
                        },
                    },
                    {
                        "item_id": 302,
                        "title": "Mechanical Keyboard",
                        "category": "Electronics",
                        "price": 149.99,
                        "predicted_rating": 4.5,
                        "confidence": 0.82,
                        "explanation_factors": {
                            "embedding_similarity": 0.78,
                            "category_preference": 0.89,
                            "price_range_fit": 0.95,
                        },
                    },
                ],
                "model_version": "3.0.0",
                "model_type": "deep_learning",
                "inference_time_ms": 0.0,
                "ab_test_variant": None,
            }
        # 4.0.0
        return {
            "user_id": user_id,
            "predictions": [
                {
                    "item_id": 401,
                    "title": "Smart Desk Lamp",
                    "category": "Home",
                    "price": 89.99,
                    "predicted_rating": 4.8,
                    "confidence": 0.93,
                    "explanation_factors": {
                        "text_similarity": 0.91,
                        "image_match": 0.87,
                        "behavioral_signal": 0.95,
                    },
                },
            ],
            "llm_explanation": (
                "Based on your recent interest in home office upgrades and positive "
                "reviews of similar smart lighting products, we think you'll love "
                "this desk lamp. It complements your recent monitor purchase."
            ),
            "model_version": "4.0.0",
            "model_type": "multimodal",
            "inference_time_ms": 0.0,
            "ab_test_variant": "multimodal_v4",
            "model_ensemble": [
                "vision_model",
                "text_model",
                "behavior_model",
                "llm_ranker",
            ],
        }

    def _log_prediction(
        self, request_version: str, model_version: str, response: dict[str, Any]
    ) -> None:
        """Log predictions for analysis and retraining."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_version": request_version,
            "model_version": model_version,
            "user_id": response["user_id"],
            "num_predictions": len(response["predictions"]),
            "top_prediction": response["predictions"][0]["item_id"]
            if response["predictions"]
            else None,
        }
        self.prediction_log.append(log_entry)

    def get_analytics(self) -> dict[str, Any]:
        """Aggregate prediction logs for analysis."""
        if not self.prediction_log:
            return {"message": "No predictions logged yet"}

        version_counts: dict[str, int] = {}
        for entry in self.prediction_log:
            version = entry["model_version"]
            version_counts[version] = version_counts.get(version, 0) + 1

        return {
            "total_predictions": len(self.prediction_log),
            "by_model_version": version_counts,
            "recent_predictions": self.prediction_log[-5:],
        }


def main() -> None:  # noqa: PLR0915
    """Runs the example."""
    print("=" * 80)
    print("ML INFERENCE PIPELINE - Schema Versioning Example")
    print("=" * 80)
    print("\nScenario: Multiple model versions deployed simultaneously")
    print("Different clients send requests in different schema versions")
    print("Service automatically migrates and routes to appropriate model\n")

    service = InferenceService()

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Legacy Client (v1.0.0)")
    print("=" * 80)
    v1_request = {
        "user_id": 12345,
        "num_recommendations": 5,
        "exclude_items": [99],
    }
    result1 = service.predict(v1_request, "1.0.0")
    print(f"\nResponse: {result1['model_version']}")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Mobile App (v2.0.0)")
    print("=" * 80)
    v2_request = {
        "user_context": {
            "user_id": 23456,
            "recent_views": [101, 102],
            "recent_purchases": [50],
            "preferred_categories": ["Electronics"],
        },
        "num_recommendations": 3,
    }
    result2 = service.predict(v2_request, "2.0.0")
    print(f"\nResponse: {result2['model_version']}")

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Web App (v3.0.0)")
    print("=" * 80)
    v3_request = {
        "user_id": 34567,
        "num_recommendations": 2,
        "diversity_weight": 0.7,
        "exploration_rate": 0.15,
    }
    result3 = service.predict(v3_request, "3.0.0")
    print(f"\nResponse: {result3['model_version']}")

    print("\n" + "=" * 80)
    print("EXAMPLE 4: A/B Test User (gets v4.0.0 model)")
    print("=" * 80)
    v3_request_ab = {
        "user_id": 40000,  # user_id % 10 == 0 triggers AB test
        "num_recommendations": 1,
    }
    result4 = service.predict(v3_request_ab, "3.0.0")
    print(f"\nResponse: {result4['model_version']}")
    if result4.get("llm_explanation"):
        print(f"LLM Explanation: {result4['llm_explanation']}")

    print("\n" + "=" * 80)
    print("ANALYTICS SUMMARY")
    print("=" * 80)
    analytics = service.get_analytics()
    print(f"\nTotal predictions: {analytics['total_predictions']}")
    print("Predictions by model version:")
    for version, count in analytics["by_model_version"].items():
        print(f"  v{version}: {count}")

    print("\n" + "=" * 80)
    print("SCHEMA EVOLUTION")
    print("=" * 80)
    diff = manager.diff("RecommendationRequest", "1.0.0", "4.0.0")
    print("\nRequest schema changes (v1.0.0 → v4.0.0):")
    print(diff.to_markdown())

    print("\n" + "=" * 80)
    print("This pattern is good for:")
    print("  ✓ ML model versioning and A/B testing")
    print("  ✓ Feature engineering evolution")
    print("  ✓ Multi-model serving")
    print("  ✓ Prediction logging for retraining")
    print("  ✓ Gradual model rollouts")
    print("=" * 80)


if __name__ == "__main__":
    main()
