import json
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from numpy import ndarray
import numpy
from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.memory.memory_store_base import MemoryStoreBase

import pinecone
from pinecone import IndexDescription
from pinecone.core.client.model.fetch_response import FetchResponse
from pinecone import Vector

MAX_DIMENSIONALITY = 20000
MAX_UPSERT_BATCH_SIZE = 100
MAX_QUERY_WITHOUT_METADATA_BATCH_SIZE = 10000
MAX_QUERY_WITH_METADATA_BATCH_SIZE = 1000
MAX_FETCH_BATCH_SIZE = 1000
MAX_DELETE_BATCH_SIZE = 1000


class PineconeMemoryStore(MemoryStoreBase):
    def __init__(self, api_key: str, environment: str, dimension: int) -> None:
        if dimension > MAX_DIMENSIONALITY:
            raise ValueError(
                f"Dimensionality of {dimension} exceeds "
                + f"the maximum allowed value of {MAX_DIMENSIONALITY}."
            )
        self.api_key = api_key
        self.environment = environment
        self.dimension = dimension
        pinecone.init(api_key=api_key, environment=environment)

    @classmethod
    def build_payload(cls, record: MemoryRecord) -> dict:
        """
        Builds a metadata payload to be sent to Pinecone from a MemoryRecord.
        """
        payload: dict = {}
        if record._text:
            payload["text"] = record._text
        if record._description:
            payload["description"] = record._description
        return payload

    @classmethod
    def parse_payload(cls, record: Vector, with_embeddings: bool) -> MemoryRecord:
        """
        Parses a record from Pinecone into a MemoryRecord.
        """
        payload = record.metadata
        description = payload.get("description", None)
        text = payload.get("text", None)
        return MemoryRecord.local_record(
            id=record.id,
            description=description,
            text=text,
            embedding=record.values if with_embeddings else numpy.array([]),
        )

    def get_collections(self) -> List[str]:
        return pinecone.list_indexes()

    async def create_collection_async(self, collection_name: str) -> None:
        """Creates a new index if it does not exist.

        Arguments:
            collection_name {str} -- The name of the index to create.

        Returns:
            None
        """
        if not await self.does_collection_exist_async(collection_name):
            pinecone.create_index(name=collection_name, dimension=self.dimension)

    async def describe_collection_async(
        self, collection_name: str
    ) -> Optional["IndexDescription"]:
        """Gets the description of the index.

        Arguments:
            collection_name {str} -- The name of the index to get.

        Returns:
            Optional[dict] -- The index.
        """
        if await self.does_collection_exist_async(collection_name):
            return pinecone.describe_index(collection_name)
        return None

    async def get_collections_async(self) -> List[str]:
        """Gets the list of indexes.

        Returns:
            List[str] -- The list of indexs.
        """
        return pinecone.list_indexes()

    async def delete_collection_async(self, collection_name: str) -> None:
        """Deletes a index.

        Arguments:
            collection_name {str} -- The name of the collection to delete.

        Returns:
            None
        """
        if await self.does_collection_exist_async(collection_name):
            pinecone.delete_index(name=collection_name)

    async def does_collection_exist_async(self, collection_name: str) -> bool:
        """Checks if a index exists.

        Arguments:
            collection_name {str} -- The name of the index to check.

        Returns:
            bool -- True if the index exists; otherwise, False.
        """
        return collection_name in await self.get_collections_async()

    async def upsert_async(self, collection_name: str, record: MemoryRecord) -> str:
        """Upserts a record.

        Arguments:
            collection_name {str} -- The name of the index to upsert the record into.
            record {MemoryRecord} -- The record to upsert.

        Returns:
            str -- The unqiue database key of the record.
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Index '{collection_name}' does not exist")

        index = pinecone.Index(collection_name)
        payload = PineconeMemoryStore.build_payload(record)
        index.upsert([(record._id, record.embedding.tolist(), payload)])
        return record._id

    async def upsert_batch_async(
        self, collection_name: str, records: List[MemoryRecord]
    ) -> List[str]:
        """Upserts a batch of records.

        Arguments:
            collection_name {str} -- The name of the collection to upsert the records into.
            records {List[MemoryRecord]} -- The records to upsert.

        Returns:
            List[str] -- The unqiue database keys of the records.
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        index = pinecone.Index(collection_name)
        vectors = [
            (
                record._id,
                record.embedding.tolist(),
                PineconeMemoryStore.build_payload(record),
            )
            for record in records
        ]
        index.upsert(vectors, batch_size=MAX_UPSERT_BATCH_SIZE)

        return [record._id for record in records]

    async def get_async(
        self, collection_name: str, key: str, with_embedding: bool
    ) -> MemoryRecord:
        """Gets a record.

        Arguments:
            collection_name {str} -- The name of the collection to get the record from.
            key {str} -- The unique database key of the record.
            with_embedding {bool} -- Whether to include the embedding in the result. (default: {False})

        Returns:
            MemoryRecord -- The record.
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        index = pinecone.Index(collection_name)
        result = index.fetch([key])

        if len(result.vectors) == 0:
            raise KeyError(f"Record with key '{key}' does not exist")

        return PineconeMemoryStore.parse_payload(result.vectors[key], with_embedding)

    async def _get_batch_async(
        self, collection_name: str, keys: List[str], with_embedding: bool
    ) -> "FetchResponse":
        index = pinecone.Index(collection_name)
        if len(keys) > MAX_FETCH_BATCH_SIZE:
            results = index.fetch(keys[0:MAX_FETCH_BATCH_SIZE])["vectors"]
            for i in range(MAX_FETCH_BATCH_SIZE, len(keys), MAX_FETCH_BATCH_SIZE):
                results.vectors.update(
                    index.fetch(keys[i : i + MAX_FETCH_BATCH_SIZE]).vectors
                )
        else:
            results = index.fetch(keys)
        return results

    async def get_batch_async(
        self, collection_name: str, keys: List[str], with_embedding: bool
    ) -> List[MemoryRecord]:
        """Gets a batch of records.

        Arguments:
            collection_name {str} -- The name of the collection to get the records from.
            keys {List[str]} -- The unique database keys of the records.
            with_embeddings {bool} -- Whether to include the embeddings in the results. (default: {False})

        Returns:
            List[MemoryRecord] -- The records.
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        result = await self._get_batch_async(collection_name, keys, with_embedding)
        return [
            PineconeMemoryStore.parse_payload(result.vectors[key], with_embedding)
            for key in result.vectors.keys()
        ]

    async def remove_async(self, collection_name: str, key: str) -> None:
        """Removes a record.

        Arguments:
            collection_name {str} -- The name of the collection to remove the record from.
            key {str} -- The unique database key of the record to remove.

        Returns:
            None
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")
        index = pinecone.Index(collection_name)
        index.delete([key])

    async def remove_batch_async(self, collection_name: str, keys: List[str]) -> None:
        """Removes a batch of records.

        Arguments:
            collection_name {str} -- The name of the collection to remove the records from.
            keys {List[str]} -- The unique database keys of the records to remove.

        Returns:
            None
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")
        index = pinecone.Index(collection_name)
        for i in range(0, len(keys), MAX_DELETE_BATCH_SIZE):
            index.delete(keys[i : i + MAX_DELETE_BATCH_SIZE])
        index.delete(keys)

    async def get_nearest_matches_async(
        self,
        collection_name: str,
        embedding: ndarray,
        limit: int,
        min_relevance_score: Optional[float],
        with_embedding: bool,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Gets the nearest matches to an embedding using cosine similarity.

        Arguments:
            collection_name {str} -- The name of the collection to get the nearest matches from.
            embedding {ndarray} -- The embedding to find the nearest matches to.
            limit {int} -- The maximum number of matches to return.
            min_relevance_score {float} -- The minimum relevance score of the matches. (default: {0.0})
            with_embeddings {bool} -- Whether to include the embeddings in the results. (default: {False})

        Returns:
            List[Tuple[MemoryRecord, float]] -- The records and their relevance scores.
        """
        if not self.does_collection_exist_async(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist")

        index = pinecone.Index(collection_name)

        if limit > MAX_QUERY_WITHOUT_METADATA_BATCH_SIZE:
            raise Exception(
                "Limit must be less than or equal to "
                + f"{MAX_QUERY_WITHOUT_METADATA_BATCH_SIZE}"
            )
        elif limit > MAX_QUERY_WITH_METADATA_BATCH_SIZE:
            response = index.query(
                vector=embedding.tolist(),
                top_k=limit,
                include_values=False,
                include_metadata=False,
            )
            keys = [match.id for match in response.matches]
            records = await self._get_batch_async(collection_name, keys, with_embedding)
            vectors = records.vectors
            matches = response.matches
            for match in matches:
                vectors[match["id"]].update(match)
            results = [vectors[key] for key in vectors.keys()]
        else:
            response = index.query(
                vector=embedding.tolist(),
                top_k=limit,
                include_vector=with_embedding,
                include_values=True,
                include_metadata=True,
            )
            results = response.matches
        if min_relevance_score is not None:
            results = [match for match in results if match.score >= min_relevance_score]
        return (
            [
                (
                    PineconeMemoryStore.parse_payload(match, with_embedding),
                    match["score"],
                )
                for match in results
            ]
            if len(results) > 0
            else []
        )

    async def get_nearest_match_async(
        self,
        collection_name: str,
        embedding: ndarray,
        min_relevance_score: float,
        with_embedding: bool,
    ) -> Tuple[MemoryRecord, float]:
        """Gets the nearest match to an embedding using cosine similarity.

        Arguments:
            collection_name {str} -- The name of the collection to get the nearest match from.
            embedding {ndarray} -- The embedding to find the nearest match to.
            min_relevance_score {float} -- The minimum relevance score of the match. (default: {0.0})
            with_embedding {bool} -- Whether to include the embedding in the result. (default: {False})

        Returns:
            Tuple[MemoryRecord, float] -- The record and the relevance score.
        """
        matches = await self.get_nearest_matches_async(
            collection_name=collection_name,
            embedding=embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embedding=with_embedding,
        )
        return matches[0]
