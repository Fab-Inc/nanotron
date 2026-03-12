"""Custom LightEval task for the pedagogy benchmark (CDPK subset)."""

from __future__ import annotations

from typing import Any

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

CHOICE_KEYS = ("A", "B", "C", "D")
OPTION_FIELD_GROUPS = (
    ("option_a", "option_b", "option_c", "option_d"),
    ("answer_a", "answer_b", "answer_c", "answer_d"),
    ("choice_a", "choice_b", "choice_c", "choice_d"),
    ("a", "b", "c", "d"),
)
ANSWER_KEY_CANDIDATES = (
    "answer",
    "correct_answer",
    "label",
    "gold",
    "target",
    "correct",
    "correct_option",
)


def _normalize_text(value: Any) -> str:
    return str(value).strip()


def _extract_choices(line: dict[str, Any]) -> list[str]:
    for keys in OPTION_FIELD_GROUPS:
        if all(k in line for k in keys):
            values = [_normalize_text(line[k]) for k in keys]
            if all(values):
                return values

    if "choices" in line and isinstance(line["choices"], list):
        values = [_normalize_text(v) for v in line["choices"]]
        if len(values) >= 2:
            return values

    if "options" in line and isinstance(line["options"], list):
        values = [_normalize_text(v) for v in line["options"]]
        if len(values) >= 2:
            return values

    return []


def _extract_question(line: dict[str, Any]) -> str:
    for key in ("question", "prompt", "stem", "query"):
        if key in line:
            text = _normalize_text(line[key])
            if text:
                return text
    return ""


def _extract_answer_value(line: dict[str, Any]) -> Any:
    for key in ANSWER_KEY_CANDIDATES:
        if key in line:
            return line[key]
    return None


def _to_gold_index(answer_value: Any, choices: list[str]) -> int:
    if answer_value is None:
        return -1

    if isinstance(answer_value, int):
        return answer_value if 0 <= answer_value < len(choices) else -1

    raw = _normalize_text(answer_value)
    if not raw:
        return -1

    upper = raw.upper()
    if upper in CHOICE_KEYS[: len(choices)]:
        return CHOICE_KEYS.index(upper)

    if raw.isdigit():
        value = int(raw)
        if 0 <= value < len(choices):
            return value
        if 1 <= value <= len(choices):
            return value - 1

    for idx, choice in enumerate(choices):
        if _normalize_text(choice).lower() == raw.lower():
            return idx

    return -1


def _parse_cdpk_row(line: dict[str, Any]) -> tuple[str, list[str], int] | None:
    question = _extract_question(line)
    choices = _extract_choices(line)
    answer_value = _extract_answer_value(line)
    gold_index = _to_gold_index(answer_value, choices)

    if not question or len(choices) < 2 or gold_index < 0:
        return None

    return question, choices, gold_index


def _build_query(question: str, choices: list[str]) -> str:
    # prompt = "The following is a pedagogy multiple-choice question.\n\n"
    # prompt += f"Question: {question}\n"
    # prompt += "".join(f"{label}. {choice}\n" for label, choice in zip(CHOICE_KEYS, choices))
    # prompt += "Answer:"
    prompt = question
    return prompt


def _build_query_with_answer_space(question: str, choices: list[str]) -> str:
    # Trailing space encourages single-token continuation for cloze scoring.
    return _build_query(question, choices) + " "


def pedagogy_cdpk_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    parsed = _parse_cdpk_row(line)
    if parsed is None:
        # Registry filter should remove malformed examples. This fallback prevents hard crashes.
        return Doc(
            task_name=task_name or "pedagogy_cdpk",
            query="Question: Invalid sample\nA. Option 1\nB. Option 2\nAnswer:",
            choices=[" A", " B"],
            gold_index=0,
            instruction="The following is a pedagogy multiple-choice question.\n\n",
        )

    question, choices, gold_index = parsed
    return Doc(
        task_name=task_name or "pedagogy_cdpk",
        query=_build_query(question, choices),
        choices=[" " + label for label in CHOICE_KEYS[: len(choices)]],
        gold_index=gold_index,
        instruction="The following is a pedagogy multiple-choice question.\n\n",
    )


def pedagogy_cdpk_filter(line: dict[str, Any]) -> bool:
    return _parse_cdpk_row(line) is not None


def pedagogy_cdpk_cloze_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    parsed = _parse_cdpk_row(line)
    if parsed is None:
        return Doc(
            task_name=task_name or "pedagogy_cdpk_cloze",
            query="Question: Invalid sample\nA. Option 1\nB. Option 2\nAnswer: ",
            choices=["A", "B"],
            gold_index=0,
            instruction="The following is a pedagogy multiple-choice question.\n\n",
        )

    question, choices, gold_index = parsed
    return Doc(
        task_name=task_name or "pedagogy_cdpk_cloze",
        query=_build_query_with_answer_space(question, choices),
        # choices=list(CHOICE_KEYS[: len(choices)]),
        choices=choices,
        gold_index=gold_index,
        # instruction="The following is a pedagogy multiple-choice question.\n\n",
    )


pedagogy_cdpk = LightevalTaskConfig(
    name="pedagogy_cdpk",
    prompt_function=pedagogy_cdpk_prompt,
    hf_repo="AI-for-Education/pedagogy-benchmark",
    hf_subset="cdpk_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metric=[Metrics.exact_match],
    stop_sequence=["\n"],
    hf_filter=pedagogy_cdpk_filter,
    version=0,
)

pedagogy_cdpk_cloze = LightevalTaskConfig(
    name="pedagogy_cdpk_cloze",
    prompt_function=pedagogy_cdpk_cloze_prompt,
    hf_repo="AI-for-Education/pedagogy-benchmark",
    hf_subset="cdpk_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm],
    stop_sequence=["\n"],
    hf_filter=pedagogy_cdpk_filter,
    version=0,
)

TASKS_TABLE = [pedagogy_cdpk, pedagogy_cdpk_cloze]
