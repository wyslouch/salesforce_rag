from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient, models


class VectorStore(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    def store_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[Union[str, int]]] = None,
    ):
        """Store embeddings with their metadata in the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter=None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors and return results with metadata."""
        pass

    @abstractmethod
    def filter_bulder(self, filters):
        """Build filtering for db"""
        pass

    @abstractmethod
    def get_stats(self):
        """Get stats about the collection."""
        pass

    # @abstractmethod
    # def delete_collection(self):
    #     """Delete the collection/index."""
    #     pass


class QdrantStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "pdf_docs",
        **kwargs: Any,
    ):
        # Initialize Qdrant client and collection name
        # if use_disk is True, the data will be stored on disk
        # otherwise, it will be stored in memory
        use_disk = kwargs.get("use_disk", False)
        disk_path = kwargs.get("disk_path", "./qdrant_data")
        collection_name = kwargs.get("collection_name", "pdf_docs")
        vector_size = kwargs.get("vector_size", 1536)

        if use_disk:
            self.client = QdrantClient(path=disk_path)
        else:
            self.client = QdrantClient(":memory:")

        self.collection_name = collection_name

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def store_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[Union[str, int]]] = None,
    ):
        if ids is None:
            ids = list(range(len(embeddings)))

        if not (len(embeddings) == len(metadata) == len(ids)):
            raise ValueError("Embeddings, metadata, and ids must be the same length.")

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    vector=embedding,
                    payload=meta,
                )
                for id, embedding, meta in zip(ids, embeddings, metadata)
            ],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filter=None,
    ) -> List[Dict[str, Any]]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            with_payload=True,
            limit=top_k,
            query_filter=filter,
        )

    def filter_bulder(self, filters):
        """Build Qdrant filter from a list of filters."""
        qdrant_filter = None

        if isinstance(filters, dict):
            filters = [filters]

        should_condition = []
        for filter_item in filters:
            field = filter_item.get("field")
            value = filter_item.get("value")

            operator = filter_item.get("operator", "eq")

            if field and value is not None:
                if operator == "eq":
                    should_condition.append(
                        models.FieldCondition(
                            key=field, match=models.MatchValue(value=value)
                        )
                    )
                elif operator == "gt":
                    should_condition.append(
                        models.FieldCondition(key=field, range=models.Range(gt=value))
                    )
                elif operator == "lt":
                    should_condition.append(
                        models.FieldCondition(key=field, range=models.Range(lt=value))
                    )
                elif operator == "gte":
                    should_condition.append(
                        models.FieldCondition(key=field, range=models.Range(gte=value))
                    )
                elif operator == "lte":
                    should_condition.append(
                        models.FieldCondition(key=field, range=models.Range(lte=value))
                    )
                elif (
                    operator == "range" and isinstance(value, list) and len(value) == 2
                ):
                    should_condition.append(
                        models.FieldCondition(
                            key=field, range=models.Range(gte=value[0], lte=value[1])
                        )
                    )

        if should_condition:
            qdrant_filter = models.Filter(should=should_condition)
        return qdrant_filter

    def get_stats(self):
        """Get statistics about the vector store collection."""
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            # Get unique sources using scroll
            result = self.client.scroll(
                collection_name=self.collection_name,
                with_payload={"source": True},
                limit=1000,
            )

            # Count unique document sources
            unique_sources = set()
            for item in result[0]:
                if item.payload and "source" in item.payload:
                    unique_sources.add(item.payload["source"])

            # Get unique fiscal years and quarters
            fiscal_years_result = self.client.scroll(
                collection_name=self.collection_name,
                with_payload={"fiscal_year": True},
                limit=1000,
            )

            fiscal_quarters_result = self.client.scroll(
                collection_name=self.collection_name,
                with_payload={"fiscal_quarter": True},
                limit=1000,
            )

            unique_years = set()
            for item in fiscal_years_result[0]:
                if item.payload and "fiscal_year" in item.payload:
                    unique_years.add(item.payload["fiscal_year"])

            unique_quarters = set()
            for item in fiscal_quarters_result[0]:
                if item.payload and "fiscal_quarter" in item.payload:
                    unique_quarters.add(item.payload["fiscal_quarter"])

            # Compile stats
            stats = {
                "total_vectors": collection_info.vectors_count,
                "document_count": len(unique_sources),
                "fiscal_years": sorted(list(unique_years)),
                "fiscal_quarters": sorted(list(unique_quarters)),
                "collection_name": self.collection_name,
            }

            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None


def get_vector_store(
    vector_store_type: str = "qdrant",
    collection_name: str = "pdf_docs",
    **kwargs: Any,
) -> VectorStore:
    """Factory function to get the vector store."""
    if vector_store_type == "qdrant":
        return QdrantStore(collection_name=collection_name, **kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")
