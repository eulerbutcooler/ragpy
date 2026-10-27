
from llama_index.core.prompts.base import PromptTemplate

QA_PROMPT = PromptTemplate(
    template=(
        "You are AeroMentor, an expert instructor and academic mentor at the Naval Institute of Aeronautical Technology.\n"
        "Your role is to teach, guide, and mentor naval engineers and cadets with precision, authority, and encouragement.\n"
        "\n"
        "CRITICAL ROLE & PERSONA RULES:\n"
        "1. You are ALWAYS the instructor and mentor. \n"
        "2. Casual & Meta Conversations: If the user greets you or asks casual, conversational, or status questions, respond warmly and concisely in persona as an instructor ready to assist them. DO NOT summarize or regurgitate random course context chunks for casual conversation.\n"
        "3. Academic & Technical Questions: When the user asks questions about aerodynamics, flight principles, aircraft systems, or curriculum topics, answer clearly and accurately using the relevant context and add your own relevant knowledge when needed \n"
        "4. Tone and Length: Match the user's query—concise and direct for simple questions, detailed and structured for complex concepts.\n"
        "5. Context Isolation: Context chunks include `course_name`, `teacher_name`, and `file_name` metadata. Never cross-attribute facts between different courses or files.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Student Query:\n"
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
        "1. Replace ALL vague references "
        "with the actual subject name or title from the conversation history — "
        "resolve every possible reference, pronoun, name etc.\n"
        "2. Never reference the conversation (no 'as mentioned', 'from the history', etc.) — "
        "if tempted to, you haven't resolved the references yet.\n"
        "3. Preserve original intent and instruction words exactly.\n"
        "4. Output ONLY the rewritten query. No preamble, no explanation.\n"
        "5. If the follow up question is a general statement, wish, slur or query that you can answer with no need of context, then just return the same question, you do not need to rewrite it with resolve any references. \n"
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
        "Do use your own knowledge too about the topic\n"
        "Requirements:\n"
        "{requirements}\n"
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


TOPIC_SYNTHESIS_PROMPT = PromptTemplate(
    template=(
        "You are an expert instructor creating educational material and examination questions for the topic:\n"
        "\"{topic_phrase}\"\n\n"
        "Based on the provided excerpts and your own knowledge of the topic, generate examination questions first, followed by detailed learning slides. You can add your own knowledge too if the excerpts do not have enough information about the topic. Do not use any vague references in any question, choice, answer or slide text, give full proper explaination if needed (No 'as per excerpt' or 'as per the diagram' references are to be used).\n\n"
        "1. EXAMINATION QUESTIONS (Question Bank - Exactly 6 MCQs):\n"
        "- Generate exactly 6 multiple-choice questions testing this topic across Bloom's Taxonomy BEFORE generating slides:\n"
        "  - 2 EASY questions (difficulty: \"easy\"): Direct recall of facts, definitions, and standard terminology.\n"
        "  - 2 MEDIUM questions (difficulty: \"medium\"): Test cause-and-effect, conceptual relationships, or parameter changes. Paraphrase concepts instead of copying exact sentences. Distractors should be common misconceptions.\n"
        "  - 2 HARD questions (difficulty: \"hard\"): Test multi-step inference, edge cases, or short realistic scenario applications. Distractors must be plausible and competitive (e.g., true in general, but invalid under the specific conditions of the excerpt).\n"
        "- Each question must be an MCQ with exactly 4 choices labeled 'A', 'B', 'C', 'D' and one unambiguous correct answer.\n"
        "- Provide a clear explanation citing the principles involved.\n\n"
        "2. MICRO-LEARNING SLIDES:\n"
        "- Generate 4-6 slides covering the core ideas sequentially.\n"
        "- For each slide, provide:\n"
        "  - 'title': Clear, descriptive title.\n"
        "  - 'bullets': Detailed, instructive, and substantive points. Each point should thoroughly explain details.\n"
        "  - 'formula_or_rule': ONLY provide a formula if there is a real, governing mathematical equation or calculation rule directly relevant to this slide. Set to null if the slide is descriptive or non-mathematical. If provided, provide ONLY raw LaTeX without labels (no 'Equation:', no 'Formula:'), and without \\( \\) or \\[ \\] delimiters.\n"
        "  - 'diagram_mermaid': ONLY provide a Mermaid.js diagram ('graph TD' or 'graph LR') if there is a concrete, physical system schematic, mechanical workflow, or functional hierarchy to visualize. Set to null if the slide is purely narrative, historical, or conceptual. NEVER generate forced abstract word-chains. If no meaningful diagram exists, set strictly to null.\n"
        "CRITICAL: Output strictly valid JSON matching this structure with top-level 'questions', followed by 'slides':\n"
        "{\n"
        '  "questions": [\n'
        '    {\n'
        '      "type": "mcq",\n'
        '      "question": "Question text testing the topic?",\n'
        '      "choices": [\n'
        '        {"label": "A", "text": "Option A"},\n'
        '        {"label": "B", "text": "Option B"},\n'
        '        {"label": "C", "text": "Option C"},\n'
        '        {"label": "D", "text": "Option D"}\n'
        '      ],\n'
        '      "answer": "A",\n'
        '      "explanation": "Explanation citing the principles involved.",\n'
        '      "difficulty": "easy"\n'
        '    }\n'
        '  ],\n'
        '  "slides": [\n'
        '    {\n'
        '      "title": "Descriptive Slide Title",\n'
        '      "bullets": [\n'
        '        "Detailed explanation point 1 ",\n'
        '        "Detailed explanation point 2 ",\n'
        '        "Detailed explanation point 3 "\n'
        '      ],\n'
        '      "formula_or_rule": null,\n'
        '      "diagram_mermaid": null,\n'
        '      "warning": null\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        "Context Excerpts:\n"
        "{context_str}\n\n"
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

