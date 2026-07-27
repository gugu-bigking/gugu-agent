import base64
import io
import math
import re

import docx2txt
import numexpr
from ddgs import DDGS
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


def web_search_func(query: str, max_results: int = 5) -> str:
    """Search the public web via DuckDuckGo and return the top results.

    Use for current events, public facts, or anything not in the internal knowledge base.

    Args:
        query: search query.
        max_results: how many results to return (default 5).

    Returns:
        Results formatted as `[N] title — snippet (url)`.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i + 1}] {r['title']} — {r['body']} ({r['href']})" for i, r in enumerate(results)
    )


web_search: BaseTool = tool(web_search_func)
web_search.name = "Web_Search"


def understand_image_func(
    image_b64: str,
    mime_type: str = "image/png",
    prompt: str = "Describe this image in detail.",
) -> str:
    """Use a vision-capable LLM (gpt-4o) to describe an uploaded image.

    The image is passed as a base64-encoded data URL along with a natural-language
    prompt. Useful when the user attaches an image and asks what's in it, or wants
    OCR, scene description, or chart reading. Requires OPENAI_API_KEY.

    Args:
        image_b64: base64-encoded image bytes (no `data:` prefix).
        mime_type: image MIME type, e.g. "image/png" or "image/jpeg".
        prompt: instruction for the vision model.

    Returns:
        The model's natural-language description of the image.
    """
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


understand_image: BaseTool = tool(understand_image_func)
understand_image.name = "Understand_Image"


_DOCUMENT_TEXT_SUFFIXES = (".md", ".txt")


def understand_document_func(file_b64: str, filename: str) -> str:
    """Extract plain text from an uploaded `.docx`, `.md`, or `.txt` file.

    Args:
        file_b64: base64-encoded file bytes (no `data:` prefix).
        filename: original filename; the extension decides how text is extracted.

    Returns:
        Extracted plain text. For markdown/text files this is the file content;
        for `.docx` it is the concatenated paragraph text via `docx2txt`.
    """
    raw = base64.b64decode(file_b64)
    lower = filename.lower()

    if lower.endswith(".docx"):
        return docx2txt.process(io.BytesIO(raw))
    if lower.endswith(_DOCUMENT_TEXT_SUFFIXES):
        return raw.decode("utf-8", errors="replace")
    raise ValueError(
        f"Unsupported file type: {filename}. Supported: .docx, .md, .txt"
    )


understand_document: BaseTool = tool(understand_document_func)
understand_document.name = "Understand_Document"
