
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
        "   Example: 'tell me more about that chapter' → "
        "'Tell me more about the Aerodynamics chapter'\n"
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
        "You generate quizzes from the provided context.\n"
        "Use outside knowledge only if the provided context is insufficient "
        "to meet the requirements.\n"
        "\n"
        "Difficulty level: {difficulty}\n"
        "- easy: Basic recall and definition questions. Straightforward single-concept answers.\n"
        "- medium: Application and understanding questions. May require connecting two concepts.\n"
        "- hard: Analysis and synthesis questions. Requires deep understanding and multi-step reasoning.\n"
        "\n"
        "Requirements:\n"
        "- Generate Exactly 12 questions total.\n"
        "- 9 must be multiple-choice (type: \"mcq\") and 3 open-ended (type: \"open_ended\").\n"
        "- For mcq questions: provide exactly 4 choices labeled \"A\", \"B\", \"C\", \"D\". "
        "Each choice has a \"label\" and a \"text\". Exactly one choice is correct. "
        "Set \"answer\" to the correct choice's label (e.g. \"A\").\n"
        "- For open_ended questions: set \"choices\" to null. "
        "Set \"answer\" to a concise reference answer that captures the key point.\n"
        "- Vary question types across easy, medium, and hard difficulty.\n"
        "- Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "\n"
        "Example output:\n"
        '{"questions": [\n'
        '  {"type": "mcq", "question": "What is X?", "choices": [\n'
        '    {"label": "A", "text": "option 1"}, {"label": "B", "text": "option 2"},\n'
        '    {"label": "C", "text": "option 3"}, {"label": "D", "text": "option 4"}\n'
        '  ], "answer": "B"},\n'
        '  {"type": "open_ended", "question": "Explain Y.", "choices": null, "answer": "Y is ..."}\n'
        ']}\n'
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Output JSON:\n"
    )
)


QUIZ_GRADING_PROMPT = PromptTemplate(
    template=(
        "You are grading a student's answer to a quiz question.\n"
        "Determine if the student's answer is semantically correct, even if phrased differently.\n"
        "Tolerate paraphrasing, synonyms, and minor inaccuracies as long as the core meaning is correct.\n"
        "\n"
        "Question: {question}\n"
        "Reference answer: {reference_answer}\n"
        "Student's answer: {user_answer}\n"
        "\n"
        "Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "Set is_correct to true if the student's answer captures the key meaning, false otherwise.\n"
        "Set score to a value between 0.0 and 1.0 reflecting how correct the answer is.\n"
        "Set explanation to a brief justification of your grading decision.\n"
        "\n"
        "Output JSON:\n"
    )
)


TITLE_PROMPT = PromptTemplate(
    template=(
        "You write a short chat title from a user's first message.\n"
        "Rules:\n"
        "- 3 to 6 words, title case.\n"
        "- Summarize the topic; do not quote the message verbatim.\n"
        "- No punctuation at the end. No quotes. No markdown.\n"
        "- Plain text only. Nothing else.\n"
        "\n"
        "First message:\n"
        "{message}\n"
        "Title:"
    )
)

