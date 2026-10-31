"""Course configuration loading.

Adapting the tutor to a new course must mean writing a new YAML file, not
changing code. Everything course-specific is read from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .digests import content_digest


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    # Case-insensitive substrings matched against raw filenames at ingestion
    # time; the first topic (in config order) with a match tags the file's
    # chunks. Files matching nothing (tests, exams, the main notes) span
    # topics and are honestly left untagged rather than guessed.
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Policy:
    socratic_on_assessed_work: bool = True
    require_citations: bool = True
    refuse_out_of_scope: bool = True
    notation_source: str = "course_notes"


@dataclass(frozen=True)
class Retrieval:
    chunk_tokens: int = 800
    chunk_overlap_tokens: int = 150
    top_k: int = 8
    # tfidf | bm25 | dense | fused. TF-IDF was the only retriever until the
    # candidates were compared on generated questions (see retrieval_eval);
    # the default keeps every course that does not name one unchanged.
    method: str = "tfidf"


@dataclass(frozen=True)
class Model:
    """Default model settings. Conditions may override id, provider and effort.

    `provider` selects the backend: "anthropic" for the hosted frontier arm,
    "local" for a self-hosted open-weights model. The `local_*` fields are read
    only by the local provider and ignored otherwise.
    """

    id: str = "claude-opus-5"
    provider: str = "anthropic"
    max_tokens: int = 16000
    effort: str = "high"

    # Local provider only. Greedy by default: unlike the hosted models, local
    # inference does expose temperature and a seed, so the open arm can be made
    # exactly reproducible and needs no variance runs.
    local_max_new_tokens: int = 1024
    local_load_in_4bit: bool = True
    local_temperature: float = 0.0
    local_seed: int = 0


@dataclass(frozen=True)
class Frontier:
    """An external frontier model for the cross-model ungrounded baseline.

    Model ids, endpoints, key names and prices all change often, and none of
    them is a design choice of this project — which is why they live here and
    not in code. Any endpoint that speaks the OpenAI chat-completions protocol
    works: OpenAI itself, DeepSeek, xAI, and Google's Gemini compatibility URL.
    """

    id: str
    model: str
    key_env: str                            # environment variable holding the key
    provider: str = "openai"                # the protocol, not the company
    base_url: str | None = None             # None = the OpenAI default endpoint
    price_in: float = 0.0                   # USD per million input tokens
    price_out: float = 0.0                  # USD per million output tokens
    max_tokens_param: str = "max_tokens"    # OpenAI's newer models want max_completion_tokens


@dataclass(frozen=True)
class CourseConfig:
    code: str
    name: str
    institution: str
    level: int
    convener: str | None
    topics: tuple[Topic, ...]
    policy: Policy
    retrieval: Retrieval
    model: Model
    frontier: tuple[Frontier, ...]
    path: Path
    # How the course runs: dates, weightings, who teaches it. These are the facts a
    # student asks about that no passage of notes contains, and that a search over
    # 800-token chunks of a handout finds barely half the time. Stated in the config,
    # they reach the prompt directly, and there is nothing left to retrieve wrongly.
    #
    # Retained as a measurable alternative, not as the delivered behaviour: the tutor
    # now answers these from `handout` below. See retrieval.passages_for.
    facts: tuple[tuple[str, str], ...] = ()
    # The document that states how the course runs, by its name in the corpus. A
    # question about the course is answered from this file and nothing else: it is
    # short enough to supply whole, so there is no ranking step to get wrong, and no
    # past paper can be returned in its place.
    handout: str = ""
    digest: str = field(compare=False, default="")

    @property
    def topic_ids(self) -> set[str]:
        return {t.id for t in self.topics}

    def topic_name(self, topic_id: str) -> str:
        for t in self.topics:
            if t.id == topic_id:
                return t.name
        return topic_id

    def frontier_by_id(self, frontier_id: str) -> Frontier:
        for f in self.frontier:
            if f.id == frontier_id:
                return f
        raise KeyError(
            f"{self.path}: no `frontier:` entry with id '{frontier_id}'. The "
            f"C0-{frontier_id} condition needs one — see the frontier block in the "
            "course config."
        )


def load_config(path: str | Path) -> CourseConfig:
    """Read a course YAML file into a CourseConfig.

    The digest is recorded in every run manifest so a result can always be traced
    back to the exact configuration that produced it.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)

    course = raw.get("course") or {}
    if not course.get("code"):
        raise ValueError(f"{path}: course.code is required")

    topics = tuple(
        Topic(
            id=t["id"],
            name=t.get("name", t["id"]),
            patterns=tuple(t.get("patterns") or ()),
        )
        for t in raw.get("topics") or []
    )
    if not topics:
        raise ValueError(f"{path}: at least one topic is required")

    frontier = tuple(
        Frontier(
            id=f["id"],
            model=f["model"],
            key_env=f["key_env"],
            provider=f.get("provider", "openai"),
            base_url=f.get("base_url"),
            price_in=float(f.get("price_in") or 0.0),
            price_out=float(f.get("price_out") or 0.0),
            max_tokens_param=f.get("max_tokens_param", "max_tokens"),
        )
        for f in raw.get("frontier") or []
    )

    return CourseConfig(
        code=course["code"],
        name=course.get("name", course["code"]),
        institution=course.get("institution", ""),
        level=int(course.get("level", 0)),
        convener=course.get("convener"),
        topics=topics,
        policy=Policy(**(raw.get("policy") or {})),
        retrieval=Retrieval(**(raw.get("retrieval") or {})),
        model=Model(**(raw.get("model") or {})),
        frontier=frontier,
        path=path,
        facts=tuple((str(k), str(v)) for k, v in (raw.get("facts") or {}).items()),
        handout=str(raw.get("handout") or ""),
        digest=content_digest(raw_bytes),
    )
