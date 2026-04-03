"""Custom LightEval task for the pedagogy benchmark (CDPK subset)."""

from __future__ import annotations

from typing import Any

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def qurating_pairs_cloze_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    def parse_line(line):
        minlen = min(len(line["high_text"]), len(line["low_text"]))
        uselen = min(minlen, 8000)
        return [line["high_text"][:uselen], line["low_text"][:uselen]]
    return Doc(
        task_name=task_name or "qurating_pairs_cloze",
        query="",
        # choices=list(CHOICE_KEYS[: len(choices)]),
        choices=parse_line(line),
        gold_index=0,
        # instruction="The following is a pedagogy multiple-choice question.\n\n",
    )
    
def qurating_pairs_filter(line: dict[str, Any]) -> bool:
    return line["low_token_count"] < 3000 and line["high_token_count"] < 3000

qurating_pairs_cloze = LightevalTaskConfig(
    name="qurating_pairs_cloze",
    prompt_function=qurating_pairs_cloze_prompt,
    hf_repo="AI-for-Education/qurating-core-edu-pairs",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm],
    hf_filter=None,#qurating_pairs_filter,
    version=0,
)

TASKS_TABLE = [qurating_pairs_cloze]
