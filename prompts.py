"""
Centralized repository of every LLM prompt used across the application.

All system messages, user-message templates, and reusable prompt fragments live
here.  No prompt text should be defined anywhere else in the codebase.

Sections
--------
1.  Chat-with-docs prompts                     (routes/rag_routes.py)
2.  Image-editing prompts — Gemini             (services/image_service.py)
3.  Image-QA detection prompts — OpenAI vision (services/image_service.py)
4.  Image-QA correction prompts — OpenAI text  (services/image_service.py)
"""

# =============================================================================
# 1.  CHAT-WITH-DOCS PROMPTS
#
#     Used by:
#       • routes/rag_routes.py → /chat-with-docs
#         state.llm.invoke([{"role": "system", ...}, {"role": "user", ...}])
#
#     CHAT_WITH_DOCS_SYSTEM  is the fixed system message that constrains the
#     assistant to answer only from supplied document context.
#
#     CHAT_WITH_DOCS_USER_TEMPLATE  is a format string for the user message;
#     call .format(chat_history=..., user_question=..., context=...) to fill it.
# =============================================================================

CHAT_WITH_DOCS_SYSTEM = (
    "You are a medical assistant that answers strictly from supplied document context."
)

CHAT_WITH_DOCS_USER_TEMPLATE = (
    "Answer the user question using only the provided document context. "
    "If the answer is not present in context, clearly say that it is not found in the selected documents. "
    "Keep answer concise and clinically accurate.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Question: {user_question}\n\n"
    "Context:\n{context}"
)


# =============================================================================
# 2.  IMAGE-EDITING PROMPTS — GEMINI
#
#     Used by:
#       • services/image_service.py → edit_image()
#         state.gemini_client.models.generate_content(contents=[prompt, image])
#
#     EDIT_IMAGE_USER_PREFIX  is the opening of every Gemini edit prompt.
#     The caller appends  f"Changes: {changes}"  then optionally
#     EDIT_VISUAL_CONTINUITY.
#
#     EDIT_VISUAL_CONTINUITY  is appended when preserve_visual_identity=True
#     (i.e. during the accuracy-pipeline correction passes) so Gemini does not
#     reframe or restyle the image while applying surgical fixes.
# =============================================================================

EDIT_IMAGE_USER_PREFIX = "Edit the following image based on the requested changes:\n\n"

EDIT_VISUAL_CONTINUITY = (
    "\n\nVISUAL CONTINUITY (mandatory):\n"
    "• Keep the same viewpoint, framing, crop, and composition as the input — "
    "do not change camera angle, zoom, or layout.\n"
    "• Preserve background, margins, canvas edges, and negative space; "
    "only alter regions the fix explicitly requires.\n"
    "• Match the existing color palette, saturation, contrast, and lighting; "
    "do not recolor, regrade, or restyle the image for a 'new' look.\n"
    "• Keep the same illustration style, line weights, fills, and shadows.\n"
    "• Make the smallest edit that fixes the issue; the result should look "
    "almost the same as the input except for the corrected details.\n"
)


# =============================================================================
# 3.  IMAGE-QA DETECTION PROMPTS — OPENAI VISION
#
#     Used by:
#       • services/image_service.py → get_accurate_image()
#         _detect_flaws_via_openai(system_prompt=..., user_prompt=..., ...)
#
#     Stage A — structural flaws (anatomy, proportions, spatial relationships):
#       STRUCTURAL_DETECTION_SYSTEM  →  system role
#       STRUCTURAL_DETECTION_USER    →  static body of the user role message
#       STRUCTURAL_DETECTION_ORIGINAL_PROMPT_SUFFIX  →  optional suffix template;
#         fill with original_prompt.strip() when available
#
#     Stage B — label & annotation flaws:
#       LABEL_DETECTION_SYSTEM   →  system role
#       LABEL_DETECTION_USER     →  static body of the user role message
#       LABEL_DETECTION_ORIGINAL_PROMPT_SUFFIX  →  optional suffix template
# =============================================================================

STRUCTURAL_DETECTION_SYSTEM = (
    "You are a rigorous medical illustration quality-control expert. "
    "Your sole job is to detect inaccuracies in the structural design of "
    "scientific diagrams (anatomy, proportions, spatial relationships, topology). "
    "You are thorough, critical, and never lenient — report every structural error, "
    "no matter how subtle."
)

STRUCTURAL_DETECTION_USER = (
    "Examine this diagram image with extreme care.\n\n"
    "STEP 1 — Inventory: Describe the overall structural design — shapes, "
    "relative positions, proportions, and connections between parts.\n\n"
    "STEP 2 — Verify the structural design against your medical/scientific knowledge:\n"
    "  • Are structures anatomically/scientifically correct in shape and form?\n"
    "  • Are sizes and proportions realistic relative to each other?\n"
    "  • Are spatial relationships and topology (what connects to what, and where) accurate?\n"
    "  • Are any components missing, duplicated, distorted, or in the wrong location?\n\n"
    "STEP 3 — Report structural flaws as a numbered list, most critical first. "
    "Each item: ONE flaw and what it should be instead.\n"
    "If there are absolutely no structural errors: output only NO_FLAWS_DETECTED."
)

# Appended to STRUCTURAL_DETECTION_USER when original_prompt is available.
# Format: STRUCTURAL_DETECTION_ORIGINAL_PROMPT_SUFFIX.format(original_prompt=...)
STRUCTURAL_DETECTION_ORIGINAL_PROMPT_SUFFIX = (
    "\n\nORIGINAL PROMPT — use this to prioritise which structural properties matter most:\n"
    "{original_prompt}"
)

LABEL_DETECTION_SYSTEM = (
    "You are a rigorous medical illustration quality-control expert specialising in "
    "label and annotation accuracy. You check spelling, anatomical correctness of each "
    "label name, arrow targets, and visual legibility of all text. "
    "You are thorough and never lenient — report every label error, however small."
)

LABEL_DETECTION_USER = (
    "Examine this diagram image with extreme care, focusing exclusively on labels, "
    "annotations, callout lines, and arrows.\n\n"
    "STEP 1 — Inventory: List every label, annotation, and arrow visible.\n\n"
    "STEP 2 — Verify each label and arrow:\n"
    "  • Is the label text spelled correctly?\n"
    "  • Does the label correctly name the structure it refers to?\n"
    "  • Is the arrow/callout pointing to the correct structure?\n"
    "  • Is the text clean, undistorted, and fully legible "
    "(no blurring, warping, overlapping, or garbling)?\n"
    "  • Are any labels missing, duplicated, or on the wrong structure?\n\n"
    "STEP 3 — Report label/annotation flaws as a numbered list, most critical first. "
    "Each item: ONE flaw, what is wrong, and what it should say or point to instead.\n"
    "If there are absolutely no label errors: output only NO_FLAWS_DETECTED."
)

# Appended to LABEL_DETECTION_USER when original_prompt is available.
# Format: LABEL_DETECTION_ORIGINAL_PROMPT_SUFFIX.format(original_prompt=...)
LABEL_DETECTION_ORIGINAL_PROMPT_SUFFIX = (
    "\n\nORIGINAL PROMPT for context:\n{original_prompt}"
)


# =============================================================================
# 4.  IMAGE-QA CORRECTION PROMPTS — OPENAI TEXT
#
#     Used by:
#       • services/image_service.py → get_accurate_image()
#         oa_client.chat.completions.create(messages=[system, user])
#
#     OpenAI is asked to translate raw flaw lists into precise Gemini edit
#     instructions.  The generated instructions are then passed to Gemini via
#     edit_image().
#
#     STRUCTURAL_CORRECTION_SYSTEM  →  system role for structural-fix pass(es)
#     LABEL_POLISH_SYSTEM           →  system role for the final label-polish pass
#
#     INTENT_SUFFIX_TEMPLATE  →  appended to the user message whenever an
#       original_prompt is available, so OpenAI can preserve the generation intent.
#       Format: INTENT_SUFFIX_TEMPLATE.format(original_prompt=...)
# =============================================================================

STRUCTURAL_CORRECTION_SYSTEM = (
    "You are an expert at writing precise image-editing instructions for "
    "AI image models. Given a list of structural flaws in a scientific diagram "
    "and the original generation intent, write a single, clear, actionable "
    "editing instruction that tells the image model exactly what to fix. "
    "Be specific about what is wrong and what the correct version should look like. "
    "Do NOT fix labels or text — structural changes only. "
    "The instruction MUST require preserving the original viewpoint, framing, "
    "composition, background, color palette, lighting, and illustration style — "
    "only surgically correct the listed structural issues with minimal visual drift. "
    "Output the instruction as plain text (no preamble, no bullet points)."
)

LABEL_POLISH_SYSTEM = (
    "You are an expert at writing precise image-editing instructions for "
    "AI image models. Given a list of label/annotation flaws in a scientific diagram "
    "and the original generation intent, write a single, clear, actionable "
    "editing instruction that tells the image model exactly what to fix. "
    "The instruction must: fix every listed label error, ensure all arrows point "
    "to the correct structures, and re-render ALL text in a clean sans-serif font "
    "with no blurring, warping, distortion, or overlapping — even if no specific "
    "label errors were found, because prior edit passes may have degraded text quality. "
    "Do NOT change any underlying structures or anatomy, viewpoint, framing, "
    "background, or overall colors — text and leader lines only unless a label fix "
    "requires a tiny local adjustment. "
    "Output the instruction as plain text (no preamble, no bullet points)."
)

# Appended to correction user messages when an original_prompt is available.
# Format: INTENT_SUFFIX_TEMPLATE.format(original_prompt=...)
INTENT_SUFFIX_TEMPLATE = (
    "\n\nORIGINAL PROMPT (preserve this intent):\n{original_prompt}"
)
