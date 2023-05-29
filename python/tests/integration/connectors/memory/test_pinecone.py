import numpy as np
import pytest
import os

import semantic_kernel as sk
from semantic_kernel.connectors.memory.pinecone import PineconeMemoryStore
from semantic_kernel.memory.memory_record import MemoryRecord

try:
    import pinecone  # noqa: F401

    pinecone_installed = True
except ImportError:
    pinecone_installed = False

pytestmark = pytest.mark.skipif(
    not pinecone_installed, reason="pinecone is not installed"
)


@pytest.fixture(scope="session")
def get_pinecone_config():
    if "Python_Integration_Tests" in os.environ:
        api_key = os.environ["Pinecone__ApiKey"]
        org_id = None
    else:
        # Load credentials from .env file
        api_key, org_id = sk.pinecone_settings_from_dot_env()

    return api_key, org_id


@pytest.fixture
def memory_record1():
    return MemoryRecord(
        id="test_id1",
        text="sample text1",
        is_reference=False,
        embedding=np.array([0.5, 0.5]),
        description="description",
        external_source_name="external source",
        timestamp="timestamp",
    )


@pytest.fixture
def memory_record2():
    return MemoryRecord(
        id="test_id2",
        text="sample text2",
        is_reference=False,
        embedding=np.array([0.25, 0.75]),
        description="description",
        external_source_name="external source",
        timestamp="timestamp",
    )


def test_constructor(get_pinecone_config):
    api_key, environment = get_pinecone_config
    memory = PineconeMemoryStore(api_key, environment, 2)
    assert memory.get_collections() is not None


@pytest.mark.asyncio
async def test_create_and_get_collection_async(get_pinecone_config):
    api_key, environment = get_pinecone_config
    memory = PineconeMemoryStore(api_key, environment, 2)

    await memory.create_collection_async("test-collection")
    result = await memory.describe_collection_async("test-collection")
    assert result is not None
    assert result.name == "test-collection"


@pytest.mark.asyncio
async def test_get_collections_async(get_pinecone_config):
    api_key, environment = get_pinecone_config
    memory = PineconeMemoryStore(api_key, environment, 2)

    await memory.create_collection_async("test-collection-a")
    # await memory.create_collection_async("test-collection-b")
    # await memory.create_collection_async("test-collection-c")
    result = await memory.get_collections_async()
    assert len(result) == 1


@pytest.mark.asyncio
async def test_delete_collection_async(get_pinecone_config):
    api_key, environment = get_pinecone_config
    memory = PineconeMemoryStore(api_key, environment, 2)

    await memory.create_collection_async("test-collection")
    await memory.delete_collection_async("test-collection")
    result = await memory.get_collections_async()
    assert len(result) == 0


@pytest.mark.asyncio
async def test_does_collection_exist_async(get_pinecone_config):
    api_key, environment = get_pinecone_config
    memory = PineconeMemoryStore(api_key, environment, 2)

    await memory.create_collection_async("test-collection")
    result = await memory.does_collection_exist_async("test-collection")
    assert result is True
