import asyncio
import os

from typing import Any

import mistune
from mistune.renderers.markdown import MarkdownRenderer

import semantic_kernel as sk
import semantic_kernel.core_skills.file_io_skill as sk_fio
import semantic_kernel.connectors.ai.open_ai as sk_oai
import semantic_kernel.connectors.memory.postgres as sk_mp

import semantic_kernel.text.text_chunker as sk_tc


async def pull_data(kernel: sk.Kernel, path: str, suffix: str) -> dict[str, list[str]]:
    # Get all files recursively from the path that have the suffix
    files = {}
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(suffix):
                full_path = os.path.join(root, filename)
                sub_path = os.path.relpath(full_path, path)
                files[sub_path] = await sk_fio.FileIOSkill().read_async(path=full_path)
    data = await chunk_files(files)
    return data


async def chunk_data(files: dict[str, str]) -> dict[str, list[str]]:
    # Chunk on md paragraphs
    data = {}
    for filename, text in files.items():
        data[filename] = []
        for chunk in sk_tc.split_markdown_paragraph([text], 200):
            data[filename].append(chunk)
    return data


async def chunk_files(files: dict[str, str]) -> dict[str, list[str]]:
    """Split a dictionary of files into chunks based on the first line of each file.

    Args:
        files (dict[str, str]): A dictionary of files, where the key is the filename and the value is the file's contents.

    Returns:
        dict[str, list[str]]: A dictionary of chunks, where the key is the filename and the value is a list of chunks.
    """
    parser = mistune.create_markdown(renderer="")
    chunked_files = {}
    for filename, contents in files.items():
        ast, state = parser.parse(contents)
        chunked_files[filename] = await chunk_file_ast(parser.parse(contents))
    return chunked_files


async def chunk_file_ast(ast_state: tuple[list[dict[str, str]], Any]) -> list[str]:
    """Split ast into chunks broken up by headers. Sequential headers are grouped together.
    
    Args:
        ast (list[dict[str,str]]): The AST to split into chunks.
        
    Returns:
    """
    md_renderer = MarkdownRenderer()
    ast, state = ast_state
    chunks = []
    chunk = []
    last_node_type = ""
    for node in ast[1:]:
        if node["type"] == "heading" and last_node_type != "heading":
            # Render the chunk
            if chunk:
                chunks.append(md_renderer.render_tokens(chunk, state))
            chunk = [node]
        else:
            chunk.append(node)
        
        last_node_type = node["type"]
    if chunk:
        chunks.append(md_renderer.render_tokens(chunk, state))
    return chunks


async def add_data_to_memory(kernel: sk.Kernel, data: dict[str, list[str]]) -> None:
    # Save to memory
    for filename, chunks in data.items():
        i = 1
        for chunk in chunks:
            await kernel.memory.save_information_async("essential_c_sharp", id=f"{filename}_{i}", text=chunk)
            i += 1


async def setup_chat_with_memory(kernel: sk.Kernel) -> tuple[sk.SKFunctionBase, sk.SKContext]:
    sk_prompt = """
    You are an expert on C# and a coauthor of the Essential C# book. Using your the excerpts from the book below,
    answer the questions asked by the user. Additionally, include a code example when applicable.
    If you don't know the answer, say so.
    
    Excerpts from book:
    {{$relevant_information}}

    Chat:
    {{$chat_history}}
    User: {{$user_input}}
    ChatBot: """.strip()

    chat_func = kernel.create_semantic_function(
        sk_prompt, max_tokens=2000, temperature=0.3
    )

    context = kernel.create_new_context()

    context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "essential_c_sharp"
    context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 0.8

    context["chat_history"] = ""

    return chat_func, context


async def chat(
    kernel: sk.Kernel, chat_func: sk.SKFunctionBase, context: sk.SKContext
) -> bool:
    try:
        user_input = input("User:> ")
        context["user_input"] = user_input
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    context["relevant_information"] = "\n\n".join([
        f"{result.text}"
        for result
        in await kernel.memory.search_async("essential_c_sharp", user_input, limit=5)
    ])

    answer = await kernel.run_async(chat_func, input_vars=context.variables)
    context["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"

    print(f"ChatBot:> {answer}")
    return True


async def main() -> None:
    input_folder = R"C:\Users\CaseySchadewitz\Downloads\ContentFeedNuget.1.1.0-477\Markdown"
    kernel = sk.Kernel()
    api_key, org_id = sk.openai_settings_from_dot_env()
    postgres_uri = sk.postgres_settings_from_dot_env()
    kernel.add_chat_service(
        "dv", sk_oai.OpenAIChatCompletion("gpt-3.5-turbo-16k-0613", api_key, org_id)
    )
    kernel.add_text_embedding_generation_service(
        "ada", sk_oai.OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id)
    )

    kernel.register_memory_store(memory_store=sk_mp.PostgresMemoryStore(postgres_uri, 1536, 1, 3))
    kernel.import_skill(sk_fio.FileIOSkill(), skill_name="file")
    kernel.import_skill(sk.core_skills.TextMemorySkill())

    #print("Pulling data from files...")
    #data = await pull_data(kernel, input_folder, ".md")

    #print("Adding data to memory...")
    #await add_data_to_memory(kernel, data)

    print("Setting up a chat (with memory!)")
    chat_func, context = await setup_chat_with_memory(kernel)

    print("Begin chatting (type 'exit' to exit):\n")
    chatting = True
    while chatting:
        chatting = await chat(kernel, chat_func, context)


if __name__ == "__main__":
    asyncio.run(main())