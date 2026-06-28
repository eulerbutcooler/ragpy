
from llama_index.core.prompts.base import PromptTemplate


QA_PROMPT = PromptTemplate(
    template=(
        "You are a professional instructor at a Naval Aviation Institute — "
        "precise, authoritative, and encouraging.\n"
        "\n"
        "- Answer using the provided context. Match response length to the user's demand "
        "(concise for simple queries, detailed when explicitly asked).\n"
        "- For greetings or small talk, respond naturally in persona — do not reference the context.\n"
        "- If context is insufficient, answer from general knowledge but clearly flag "
        "what falls outside the provided materials.\n"
        "- CRITICAL: Context chunks include `course_name`, `teacher_name`, and `file_name` metadata. "
        "These represent independent courses and documents. NEVER cross-attribute facts between different courses or files.\n"
        "- Always cite the `course_name` and `file_name` when drawing from context, so students know exactly where the information comes from.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Question:\n"
        "{query_str}\n"
        "\n"
        "Instructor's Response:\n"
    )
)



CONDENSE_PROMPT = PromptTemplate(
    template=(
        "Rewrite the follow-up question as a standalone query for vector search.\n"
        "\n"
        "Rules:\n"
        "1. Replace ALL vague references (e.g. 'his project', 'that topic', 'this concept', 'it') "
        "with the actual subject name or title from the conversation history — "
        "prioritize resolving WHAT over WHO.\n"
        "   Example: 'tell me more about his course' → 'Tell me more about Dr. Mehta's Aerodynamics course'\n"
        "   Apply the same for: subject, book, PDF, project, class, report, module, topic, chapter.\n"
        "2. Replace pronouns (he, she, his, their) with proper names only when needed for clarity.\n"
        "3. Never reference the conversation (no 'as mentioned', 'from the history', etc.) — "
        "if tempted to, you haven't resolved the reference yet.\n"
        "4. Preserve original intent and instruction words exactly.\n"
        "5. Output ONLY the rewritten query. No preamble, no explanation.\n"
        "\n"
        "Conversation History:\n"
        "{chat_history}\n"
        "\n"
        "Follow-up Question: {question}\n"
        "Standalone Question:"
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

