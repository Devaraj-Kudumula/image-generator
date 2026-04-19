"""
Shared helpers for /generate-prompt: exam focus, strict vs flexible grounding,
and consistent user messages for the prompt-synthesis LLM.
"""
from typing import Optional

_VALID_MODES = frozenset({"flexible", "strict"})
_VALID_EXAM = frozenset({"general", "step1", "step2"})

EXAM_FOCUS_BANNER = {
    "general": (
        "Exam focus: general medical education (not USMLE-specific unless the user asks)."
    ),
    "step1": (
        "Exam focus: USMLE Step 1 — emphasize mechanisms, associations, pathways, "
        "anatomy/histology/biochemistry relationships, and high-yield discriminators; "
        "choose figure types (pathway, schematic, comparison) that make the tested concept obvious."
    ),
    "step2": (
        "Exam focus: USMLE Step 2 CK — emphasize classic presentations, key findings, "
        "branch points between diagnoses, and decision-relevant discriminating features; "
        "use clinical-schematic or algorithm-style layouts when they clarify reasoning."
    ),
}


def normalize_prompt_mode(raw: object) -> str:
    s = str(raw or "flexible").strip().lower()
    return s if s in _VALID_MODES else "flexible"


def normalize_exam_focus(raw: object) -> str:
    s = str(raw or "general").strip().lower().replace(" ", "_")
    aliases_step1 = {"step_1", "usmle1", "usmle_step_1", "step1"}
    aliases_step2 = {"step_2", "step2_ck", "usmle2", "usmle_step_2", "step2"}
    if s in aliases_step1:
        return "step1"
    if s in aliases_step2:
        return "step2"
    return s if s in _VALID_EXAM else "general"


def exam_focus_banner(exam_focus: str) -> str:
    return EXAM_FOCUS_BANNER.get(exam_focus, EXAM_FOCUS_BANNER["general"])


def _teaching_block(teaching_notes: Optional[str]) -> str:
    if not teaching_notes or not str(teaching_notes).strip():
        return ""
    return (
        "\n\nLearner notes (integrate faithfully into the image brief; use for visual hierarchy "
        "and what must dominate the figure):\n"
        f"{str(teaching_notes).strip()}"
    )


def _strict_grounding_block(prompt_mode: str, has_rag_context: bool) -> str:
    if prompt_mode != "strict":
        return ""
    if has_rag_context:
        return (
            "\n\nSTRICT GROUNDING — REQUIRED:\n"
            "• Every factual anatomical, pathophysiological, and clinical claim in your output MUST be "
            "supported by the Retrieved High-Yield Medical Context below.\n"
            "• Do not invent structures, mechanisms, labels, or associations not present in that context. "
            "If information is missing, omit it rather than guessing.\n"
        )
    return (
        "\n\nSTRICT GROUNDING — NO RETRIEVED DOCUMENTS:\n"
        "• Ground factual claims ONLY in the user question and learner notes above.\n"
        "• Do not add anatomical or clinical detail not clearly supported by that text.\n"
    )


def _opening_structure_block(prompt_mode: str) -> str:
    if prompt_mode == "strict":
        return (
            "\n\nFINAL IMAGE PROMPT STRUCTURE:\n"
            "• The first 2–3 sentences of the image-generation prompt you output MUST state the primary "
            "teaching point and what must visually dominate the canvas, using ONLY information allowed by the "
            "strict grounding rules above.\n"
            "• Secondary content must be visibly subordinate.\n"
        )
    return (
        "\n\nFINAL IMAGE PROMPT STRUCTURE — REQUIRED FOR GEMINI:\n"
        "• The first 2–3 sentences of the image-generation prompt you write MUST state: (1) the single primary "
        "teaching point for this figure (aligned with the exam focus above), and (2) exactly what must DOMINATE "
        "the canvas (largest, center, or most salient). Secondary elements must be visibly subordinate.\n"
        "• Then specify composition, labels, arrows, and style per the system instruction.\n"
    )


def build_direct_prompt_user_message(
    structured_query: str,
    user_question: str,
    exam_focus: str,
    teaching_notes: Optional[str],
    prompt_mode: str,
) -> str:
    """User message when synthesizing a prompt without RAG context blocks."""
    lines = [
        "Create a detailed medical illustration prompt using this structured clinical query "
        "and the original request.",
        "",
        exam_focus_banner(exam_focus),
        _teaching_block(teaching_notes),
        _strict_grounding_block(prompt_mode, has_rag_context=False),
        _opening_structure_block(prompt_mode),
        "",
        f"Structured Clinical Query: {structured_query}",
        f"Original User Question: {user_question}",
    ]
    return "\n".join(lines)


def build_rag_construction_user_message(
    context: str,
    structured_query: str,
    user_question: str,
    exam_focus: str,
    teaching_notes: Optional[str],
    prompt_mode: str,
) -> str:
    """User message when retrieved context is available."""
    lines = [
        exam_focus_banner(exam_focus),
        _teaching_block(teaching_notes),
        _strict_grounding_block(prompt_mode, has_rag_context=True),
        _opening_structure_block(prompt_mode),
        "",
        "Retrieved High-Yield Medical Context:",
        context,
        "",
        "Structured Clinical Query:",
        structured_query,
        "",
        "Original User Question:",
        user_question,
        "",
        "Return a complete structured and detailed image generation prompt following the "
        "system instruction guidelines.",
    ]
    return "\n".join(lines)
