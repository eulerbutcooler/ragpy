
from llama_index.core.prompts.base import PromptTemplate


QA_PROMPT = PromptTemplate(
    template=(
        "You are a helpful, conversational AI teaching assistant.\n"
        "Use the provided context to answer the user's question accurately.\n"
        "If the user is just saying hello or making small talk, respond conversationally normally without mentioning the context.\n"
        "If the user asks a question and the context is insufficient to answer it, try your best to answer using your general intelligence.\n"
        "Cite sources by file name from the context metadata when using the context.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Question:\n"
        "{query_str}\n"
        "\n"
        "Answer:\n"
    )
)


CONDENSE_PROMPT = PromptTemplate(
    template=(
        "You are an expert query rewriter for a search engine.\n"
        "Your task is to analyze the conversation history and the latest user message, "
        "and rewrite the user's message into a single, highly specific, standalone search query.\n\n"
        "RULES:\n"
        "1. Resolve all pronouns (it, they, he, she, this, that) using the conversation history.\n"
        "2. If the user's message is a greeting (e.g., 'hi', 'hello'), a simple acknowledgment ('ok', 'thanks'), or off-topic, just return the user's message exactly as is.\n"
        "3. DO NOT answer the question. Your only job is to rewrite it.\n"
        "4. DO NOT add any conversational filler (e.g., NEVER start with 'Here is the query:' or 'The rewritten question is:'). Output ONLY the rewritten string.\n\n"
        "Conversation History:\n"
        "{chat_history}\n\n"
        "Latest User Message: {question}\n\n"
        "Standalone Search Query: "
    )
)



QUIZ_GENERATION_PROMPT = PromptTemplate(
    template=(
        "You generate quizzes from the provided context only.\n"
        "Do not use outside knowledge. If the context is insufficient, output an empty questions list.\n"
        "\n"
        "Difficulty level: {difficulty}\n"
        "- easy: Basic recall and definition questions. Straightforward single-concept answers.\n"
        "- medium: Application and understanding questions. May require connecting two concepts.\n"
        "- hard: Analysis and synthesis questions. Requires deep understanding and multi-step reasoning.\n"
        "\n"
        "Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Output JSON:\n"
    )
)

