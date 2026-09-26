"""
Structured access methods for VLM judge NLP feedback.

EvaluationResult.feedback is a flat list[str] from the judge.  This module
parses those strings into typed categories and exposes query helpers so the
orchestrator can make policy decisions without string-matching in hot paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from .evaluator import EvaluationResult


class FeedbackCategory(str, Enum):
    TOPOLOGY    = "topology"
    TEXTURE     = "texture"
    SHAPE       = "shape"
    LIGHTING    = "lighting"
    DEPTH       = "depth"
    UV          = "uv"
    SCALE       = "scale"
    UNKNOWN     = "unknown"


# Keywords that map a feedback string to a category
_CATEGORY_PATTERNS: list[tuple[FeedbackCategory, re.Pattern]] = [
    (FeedbackCategory.TOPOLOGY, re.compile(r"topolog|manifold|non.manifold|hole|face|vert|edge|tri|quad|polygon|mesh", re.I)),
    (FeedbackCategory.TEXTURE,  re.compile(r"texture|material|albedo|normal.map|roughness|metallic|color|seam|uv.stretch", re.I)),
    (FeedbackCategory.UV,       re.compile(r"\buv\b|unwrap|uv.island|uv.overlap", re.I)),
    (FeedbackCategory.SHAPE,    re.compile(r"shape|silhouette|contour|proportion|form|curve|surface", re.I)),
    (FeedbackCategory.DEPTH,    re.compile(r"depth|z.pass|parallax|relief|bas.relief", re.I)),
    (FeedbackCategory.LIGHTING, re.compile(r"light|shadow|specular|shading|ambient|occlusion|highlight", re.I)),
    (FeedbackCategory.SCALE,    re.compile(r"scale|size|dimension|aspect.ratio|too.small|too.large", re.I)),
]


@dataclass
class FeedbackItem:
    raw: str
    category: FeedbackCategory
    severity: float        # 0.0 (minor note) – 1.0 (blocking issue)
    actionable: bool       # True when the text contains a concrete suggestion


def _classify(text: str) -> FeedbackCategory:
    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(text):
            return cat
    return FeedbackCategory.UNKNOWN


def _estimate_severity(text: str) -> float:
    """Heuristic: count negative signal words and cap at 1.0."""
    high_words = re.findall(r"\b(critical|severe|major|broken|completely|missing|wrong|incorrect)\b", text, re.I)
    mid_words  = re.findall(r"\b(should|improve|needs|consider|too|slightly|minor|small)\b", text, re.I)
    score = 0.3 * len(high_words) + 0.15 * len(mid_words)
    return min(score, 1.0) or 0.2   # default 0.2 so nothing reads as zero


def _is_actionable(text: str) -> bool:
    return bool(re.search(r"\b(try|use|apply|add|remove|increase|decrease|fix|adjust|consider|should|reduce|smooth|subdivide|unwrap|retexture|re.render)\b", text, re.I))


class FeedbackAccessor:
    """
    Wraps an EvaluationResult and provides typed access to the VLM judge's
    natural-language feedback.

    Example:
        fa = FeedbackAccessor(result)
        for item in fa.by_category(FeedbackCategory.TOPOLOGY):
            print(item.raw, item.severity)
        if fa.has_blocking_issues():
            print(fa.highest_severity_item().raw)
    """

    def __init__(self, result: EvaluationResult):
        self._result = result
        self._items: list[FeedbackItem] = [
            FeedbackItem(
                raw=text,
                category=_classify(text),
                severity=_estimate_severity(text),
                actionable=_is_actionable(text),
            )
            for text in result.feedback
        ]

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    @property
    def all(self) -> list[FeedbackItem]:
        return list(self._items)

    @property
    def actionable(self) -> list[FeedbackItem]:
        return [i for i in self._items if i.actionable]

    def by_category(self, category: FeedbackCategory) -> list[FeedbackItem]:
        return [i for i in self._items if i.category == category]

    def above_severity(self, threshold: float) -> list[FeedbackItem]:
        return [i for i in self._items if i.severity >= threshold]

    def has_blocking_issues(self, threshold: float = 0.7) -> bool:
        return any(i.severity >= threshold for i in self._items)

    def highest_severity_item(self) -> FeedbackItem | None:
        return max(self._items, key=lambda i: i.severity, default=None)

    def priority_order(self) -> list[FeedbackItem]:
        """Items sorted highest-severity first, actionable items promoted."""
        return sorted(self._items, key=lambda i: (i.severity + (0.1 if i.actionable else 0)), reverse=True)

    def categories_present(self) -> set[FeedbackCategory]:
        return {i.category for i in self._items}

    def summary(self) -> str:
        lines = [f"Score: {self._result.overall_score:.2f}"]
        for item in self.priority_order()[:5]:
            tag = f"[{item.category.value}|{item.severity:.1f}]"
            lines.append(f"  {tag} {item.raw}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Iteration support
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[FeedbackItem]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
