def compose_scene_text_condition(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return f"[Scene:]{text}"


def compose_event_text_condition(text: str, text_edit: str) -> str:
    text = (text or "").strip()
    text_edit = (text_edit or "").strip()
    parts = []
    if text:
        parts.append(f"[Scene:]{text}")
    if text_edit:
        parts.append(f"[Event:]{text_edit}")
    return "".join(parts)

def compose_instruction_text_condition(text: str, text_instruction: str) -> str:
    text = (text or "").strip()
    text_instruction = (text_instruction or "").strip()
    parts = []
    if text:
        parts.append(f"[Scene:]{text}")
    if text_instruction:
        parts.append(f"[Instruction:]{text_instruction}")
    return "".join(parts)


def compose_chunk_text_condition(
    text_edit: str, text_instruction: str, state: str, use_instruction: bool
) -> str:
    """Per-chunk prompt a_i = [Instruction:]/[Event:] + [State:], WITHOUT the scene.

    Scene is carried separately by the background prompt a_B and the global
    prompt a_G; a_i only holds the chunk-local instruction/event plus its state.
    """
    parts = []
    if use_instruction:
        ti = (text_instruction or "").strip()
        if ti:
            parts.append(f"[Instruction:]{ti}")
    else:
        te = (text_edit or "").strip()
        if te:
            parts.append(f"[Event:]{te}")
    st = (state or "").strip()
    if st:
        parts.append(f"[State:]{st.capitalize()}")
    return "".join(parts)
