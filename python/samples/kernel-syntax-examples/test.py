import asyncio
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
import semantic_kernel.memory.volatile_memory_store as sk_mv


async def test():
    # Create a new kernel
    kernel = sk.Kernel()

    # Regiter openai embedding service
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_embedding_generation_service(
        "ada", sk_oai.OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id)
    )

    # Create a new memory store
    kernel.register_memory_store(sk_mv.VolatileMemoryStore())

    await kernel.memory.save_information_async("test", id="test1", text="hello world")

if __name__ == "__main__":
    asyncio.run(test())

