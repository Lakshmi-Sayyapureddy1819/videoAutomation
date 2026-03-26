from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from dotenv import load_dotenv

from scout import get_best_visual_source, search_multi_source_candidates

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

load_dotenv()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "cache")
_EMBED_CACHE_PATH = os.path.join(_CACHE_DIR, "embeddings.json")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "he", "her", "his",
    "in", "into", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "with",
}
PRONOUN_PREFIXES = ("he ", "she ", "they ", "his ", "her ", "their ", "it ")
HISTORICAL_MARKERS = (
    "ancient", "medieval", "victorian", "colonial", "ww1", "ww2", "tsar", "empire",
    "dynasty", "century", "historical", "royal",
)
MODERN_NEGATIVE_HINTS = [
    "modern scenes",
    "contemporary clothing",
    "smartphones",
    "city traffic",
    "unrelated people",
]


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_json_payload(text: str) -> Any:
    if not text:
        raise ValueError("Empty LLM response")
    content = text.strip()
    if "```" in content:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            content = blocks[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _unique_preserve(values):
    result = []
    seen = set()
    for value in values:
        cleaned = _normalize_space(str(value))
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def _coerce_list(value):
    if isinstance(value, list):
        return _unique_preserve([item for item in value if item])
    if isinstance(value, str) and value.strip():
        return _unique_preserve(re.split(r"[,|;/]", value))
    return []


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _coerce_int(value, default=5):
    try:
        return max(1, min(10, int(value)))
    except Exception:
        return default


def _is_historical_text(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in HISTORICAL_MARKERS) or bool(
        re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", lowered)
    )


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9']+", text.lower()) if token and token not in STOP_WORDS]


def _extract_keywords(text: str, limit=6) -> list[str]:
    tokens = _tokenize(text)
    result = []
    seen = set()
    for token in tokens:
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result[:limit]


def _extract_entity_aliases(entity: str, keywords: list[str] | None = None) -> list[str]:
    aliases = []
    if entity:
        aliases.append(entity)
        parts = [part for part in entity.split() if len(part) > 2]
        if len(parts) > 1:
            aliases.append(parts[-1])
        aliases.extend(parts)
    aliases.extend(keywords or [])
    return _unique_preserve(aliases)


def _candidate_mentions_entity(candidate: dict[str, Any], entity: str, keywords: list[str] | None = None) -> bool:
    if not entity:
        return False
    haystack = " ".join(
        [candidate.get("title", ""), candidate.get("description", ""), " ".join(candidate.get("tags", []) or [])]
    ).lower()
    aliases = _extract_entity_aliases(entity, keywords)
    return any(alias.lower() in haystack for alias in aliases if len(alias) > 2)


def _keyword_overlap_score(scene_keywords: list[str], candidate_text: str) -> float:
    if not scene_keywords:
        return 0.0
    candidate_tokens = set(_tokenize(candidate_text))
    overlap = sum(1 for keyword in scene_keywords if keyword.lower() in candidate_tokens or keyword.lower() in candidate_text.lower())
    return overlap / max(1, len(scene_keywords))


def _extract_time_period(text: str, inherited: str | None = None) -> str | None:
    year_match = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text)
    if year_match:
        return year_match.group(1)
    century_match = re.search(r"\b(\d{1,2}(?:st|nd|rd|th)\s+century)\b", text.lower())
    if century_match:
        return century_match.group(1)
    if _is_historical_text(text):
        return inherited or "historical period"
    return inherited


def _extract_location(text: str, inherited: str | None = None) -> str | None:
    patterns = [
        r"\bin ([A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)*)",
        r"\bfrom ([A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)*)",
        r"\bat ([A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return inherited


def _extract_entity(sentence: str, inherited_entity: str | None = None) -> str | None:
    cleaned = _normalize_space(sentence)
    lowered = cleaned.lower()
    if inherited_entity and lowered.startswith(PRONOUN_PREFIXES):
        return inherited_entity
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", cleaned)
    filtered = [match for match in matches if match.lower() not in {"the", "a", "an", "he", "she", "they"}]
    return filtered[0] if filtered else inherited_entity


def _infer_entity_type(entity: str | None, sentence: str) -> str:
    lowered = sentence.lower()
    if any(word in lowered for word in ["war", "battle", "revolution", "election", "coronation"]):
        return "event"
    if any(word in lowered for word in ["city", "village", "country", "river", "palace", "mountain"]):
        return "place"
    if entity:
        return "person"
    if any(word in lowered for word in ["idea", "belief", "concept", "system"]):
        return "concept"
    return "topic"


def _infer_context_type(sentence: str) -> str:
    lowered = sentence.lower()
    if any(word in lowered for word in ["felt", "fear", "hope", "anger", "sad", "joy"]):
        return "emotion"
    if any(word in lowered for word in ["was born", "grew up", "family", "childhood", "came from"]):
        return "biography"
    if any(word in lowered for word in ["walked", "fought", "built", "entered", "led", "moved"]):
        return "action"
    if any(word in lowered for word in ["looked", "appeared", "described", "covered"]):
        return "description"
    return "narration"


def _infer_mood(sentence: str) -> str:
    lowered = sentence.lower()
    if any(word in lowered for word in ["mysterious", "dramatic", "power", "dark", "ominous"]):
        return "dramatic"
    if any(word in lowered for word in ["hope", "vision", "inspired", "freedom"]):
        return "inspirational"
    if any(word in lowered for word in ["war", "death", "violence", "tragedy"]):
        return "serious"
    return "neutral"


def _heuristic_scene(sentence: str, context: "GlobalContextMemory") -> dict[str, Any]:
    main_entity = _extract_entity(sentence, context.current_entity)
    entity_type = _infer_entity_type(main_entity, sentence)
    time_period = _extract_time_period(sentence, context.current_time)
    location = _extract_location(sentence, context.current_location)
    keywords = _unique_preserve(([main_entity] if main_entity else []) + _extract_keywords(sentence))
    strict = bool(main_entity and (sentence.lower().startswith(PRONOUN_PREFIXES) or entity_type in {"person", "event", "place"}))
    visual_intent = sentence if not main_entity or main_entity in sentence else f"{main_entity} {sentence}"
    negative_keywords = ["random generic visuals", "misleading unrelated footage"]
    if time_period and _is_historical_text(time_period):
        negative_keywords.extend(MODERN_NEGATIVE_HINTS)
    return {
        "sentence": sentence,
        "main_entity": main_entity or "",
        "entity_type": entity_type,
        "secondary_entities": [],
        "context_type": _infer_context_type(sentence),
        "time_period": time_period or "",
        "location": location or "",
        "visual_intent": visual_intent,
        "mood": _infer_mood(sentence),
        "keywords": keywords or _extract_keywords(sentence),
        "negative_keywords": _unique_preserve(negative_keywords),
        "strict_entity_required": strict,
        "importance_score": 8 if strict else 6,
    }


@dataclass
class GlobalContextMemory:
    current_entity: str | None = None
    current_time: str | None = None
    current_location: str | None = None
    visual_style: str = "documentary"

    def as_prompt_dict(self):
        return {
            "current_entity": self.current_entity,
            "current_time": self.current_time,
            "current_location": self.current_location,
            "visual_style": self.visual_style,
        }

    def absorb(self, scene: "SceneAnalysis"):
        if scene.main_entity:
            self.current_entity = scene.main_entity
        if scene.time_period:
            self.current_time = scene.time_period
        if scene.location:
            self.current_location = scene.location


@dataclass
class SceneAnalysis:
    sentence: str
    main_entity: str = ""
    entity_type: str = "topic"
    secondary_entities: list[str] = field(default_factory=list)
    context_type: str = "narration"
    time_period: str = ""
    location: str = ""
    visual_intent: str = ""
    mood: str = "neutral"
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    strict_entity_required: bool = False
    importance_score: int = 5
    search_queries: list[str] = field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    rerank_reason: str = ""
    llm_score: float = 0.0
    validation_passed: bool = False
    fallback_used: bool = False
    clip_prompt: str = ""

    def to_dict(self):
        return asdict(self)


class EmbeddingService:
    def __init__(self):
        _ensure_cache_dir()
        self.cache = self._load_cache()
        self.backend = None
        self.dimension = None
        self.model = None
        self.client = None

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.backend = "sentence-transformers"
        except Exception:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI

                    self.client = OpenAI(api_key=api_key)
                    self.backend = "openai"
                    self.dimension = 1536
                except Exception:
                    self.client = None

    def _load_cache(self):
        if not os.path.exists(_EMBED_CACHE_PATH):
            return {}
        try:
            with open(_EMBED_CACHE_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _save_cache(self):
        try:
            with open(_EMBED_CACHE_PATH, "w", encoding="utf-8") as handle:
                json.dump(self.cache, handle)
        except Exception:
            pass

    def _cache_key(self, text):
        return re.sub(r"\s+", " ", text.strip().lower())[:4000]

    def encode_many(self, texts: list[str]) -> np.ndarray | None:
        cleaned = [_normalize_space(text) for text in texts]
        if not cleaned or not self.backend:
            return None

        vectors = [None] * len(cleaned)
        missing_indices = []
        missing_texts = []
        for index, text in enumerate(cleaned):
            key = self._cache_key(text)
            if key in self.cache:
                vectors[index] = np.asarray(self.cache[key], dtype=np.float32)
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        if missing_texts:
            if self.backend == "sentence-transformers":
                encoded = self.model.encode(missing_texts, normalize_embeddings=True)
            else:
                response = self.client.embeddings.create(model="text-embedding-3-small", input=missing_texts)
                encoded = [item.embedding for item in response.data]

            for index, vector in zip(missing_indices, encoded):
                arr = np.asarray(vector, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vectors[index] = arr
                self.cache[self._cache_key(cleaned[index])] = arr.tolist()
            self._save_cache()

        return np.vstack(vectors).astype(np.float32)


class ScriptIntelligenceEngine:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

    def segment_text(self, text: str, preferred_scene_count: int | None = None) -> list[str]:
        normalized = _normalize_space((text or "").replace("\n", " "))
        if not normalized:
            return []

        raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
        segments = []
        for sentence in raw_sentences:
            sentence = _normalize_space(sentence)
            if not sentence:
                continue
            if len(sentence) > 220 or sentence.count(",") >= 3 or sentence.count(";") >= 1:
                sub_parts = re.split(r"(?<=[,;:])\s+|\s+(?:and|but|while)\s+", sentence)
                sub_parts = [_normalize_space(part) for part in sub_parts if len(_normalize_space(part)) > 20]
                segments.extend(sub_parts if len(sub_parts) > 1 else [sentence])
            else:
                segments.append(sentence)

        if preferred_scene_count and preferred_scene_count > 0 and len(segments) > preferred_scene_count * 2:
            grouped = []
            group_size = int(math.ceil(len(segments) / preferred_scene_count))
            for index in range(0, len(segments), group_size):
                grouped.append(_normalize_space(" ".join(segments[index:index + group_size])))
            segments = grouped

        return [segment for segment in segments if segment]

    def analyze_sentence(self, sentence: str, context: GlobalContextMemory) -> SceneAnalysis:
        fallback = _heuristic_scene(sentence, context)
        payload = fallback

        if self.client:
            prompt = f"""
You are an entity-aware documentary scene planner.
Resolve pronouns using the provided global context.
Return only one JSON object with these fields:
sentence, main_entity, entity_type, secondary_entities, context_type, time_period, location,
visual_intent, mood, keywords, negative_keywords, strict_entity_required, importance_score.

Rules:
- If the sentence depends on a specific person/place/event, keep strict_entity_required=true.
- If the sentence is symbolic or generic, strict_entity_required=false.
- Preserve historical continuity from context unless the sentence clearly changes era.
- Keywords must be strong search terms, not a copy of the full sentence.
- Negative keywords should prevent wrong visuals.

Global context:
{json.dumps(context.as_prompt_dict(), ensure_ascii=False)}

Sentence:
{sentence}
""".strip()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                parsed = _extract_json_payload(response.choices[0].message.content or "")
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = fallback

        scene = SceneAnalysis(
            sentence=_normalize_space(payload.get("sentence") or sentence),
            main_entity=_normalize_space(payload.get("main_entity", "")),
            entity_type=_normalize_space(payload.get("entity_type", "")) or fallback["entity_type"],
            secondary_entities=_coerce_list(payload.get("secondary_entities")),
            context_type=_normalize_space(payload.get("context_type", "")) or fallback["context_type"],
            time_period=_normalize_space(payload.get("time_period", "")),
            location=_normalize_space(payload.get("location", "")),
            visual_intent=_normalize_space(payload.get("visual_intent", "")) or fallback["visual_intent"],
            mood=_normalize_space(payload.get("mood", "")) or fallback["mood"],
            keywords=_coerce_list(payload.get("keywords")) or fallback["keywords"],
            negative_keywords=_coerce_list(payload.get("negative_keywords")) or fallback["negative_keywords"],
            strict_entity_required=_coerce_bool(payload.get("strict_entity_required", fallback["strict_entity_required"])),
            importance_score=_coerce_int(payload.get("importance_score"), default=fallback["importance_score"]),
        )

        lowered = scene.sentence.lower()
        if context.current_entity and lowered.startswith(PRONOUN_PREFIXES) and not scene.main_entity:
            scene.main_entity = context.current_entity
        if context.current_entity and lowered.startswith(PRONOUN_PREFIXES):
            scene.strict_entity_required = True
        if not scene.time_period and context.current_time:
            scene.time_period = context.current_time
        if not scene.location and context.current_location:
            scene.location = context.current_location
        if scene.time_period and _is_historical_text(scene.time_period):
            scene.negative_keywords = _unique_preserve(scene.negative_keywords + MODERN_NEGATIVE_HINTS)
        scene.keywords = _unique_preserve(([scene.main_entity] if scene.main_entity else []) + scene.keywords)
        scene.clip_prompt = self.build_clip_prompt(scene)
        return scene

    def build_clip_prompt(self, scene: SceneAnalysis) -> str:
        parts = [
            scene.main_entity,
            scene.visual_intent,
            scene.location,
            scene.time_period,
            scene.mood,
            "documentary footage",
        ]
        return _normalize_space(", ".join(part for part in parts if part))


class CandidateSelector:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.embedding_service = EmbeddingService()
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

    def generate_queries(self, scene: SceneAnalysis, count=5) -> list[str]:
        entity = scene.main_entity if scene.strict_entity_required else ""
        core_terms = _unique_preserve(scene.keywords + scene.secondary_entities)
        query_candidates = [
            " ".join(part for part in [entity, core_terms[0] if core_terms else "", scene.context_type, scene.location, scene.time_period] if part),
            " ".join(part for part in [entity, scene.visual_intent, scene.location, scene.time_period] if part),
            " ".join(part for part in [entity, " ".join(core_terms[:3]), scene.time_period, "documentary footage"] if part),
            " ".join(part for part in [scene.main_entity or "", scene.location, scene.time_period, scene.mood, "historical footage"] if part),
            " ".join(part for part in [scene.visual_intent, scene.location, scene.time_period, "archive footage"] if part),
        ]
        scene.search_queries = _unique_preserve([_normalize_space(query) for query in query_candidates if _normalize_space(query)])[:count]
        return scene.search_queries

    def _semantic_rank(self, scene: SceneAnalysis, candidates: list[dict[str, Any]], top_k=15):
        query_text = _normalize_space(
            " ".join(
                part
                for part in [
                    scene.sentence,
                    scene.main_entity,
                    scene.visual_intent,
                    scene.location,
                    scene.time_period,
                    " ".join(scene.keywords),
                ]
                if part
            )
        )
        candidate_texts = [candidate.get("metadata_text", "") for candidate in candidates]
        embeddings = self.embedding_service.encode_many([query_text] + candidate_texts)
        if embeddings is None:
            scored = []
            for candidate in candidates:
                overlap = _keyword_overlap_score(scene.keywords, candidate.get("metadata_text", ""))
                scored.append((candidate, overlap))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_k]

        query_embedding = embeddings[:1]
        candidate_embeddings = embeddings[1:]
        if faiss is not None:
            index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
            index.add(candidate_embeddings)
            distances, indices = index.search(query_embedding, min(top_k, len(candidates)))
            return [(candidates[idx], float(score)) for score, idx in zip(distances[0], indices[0]) if idx != -1]

        scores = candidate_embeddings @ query_embedding.T
        flat_scores = scores.reshape(-1)
        order = np.argsort(flat_scores)[::-1][:top_k]
        return [(candidates[idx], float(flat_scores[idx])) for idx in order]

    def _heuristic_llm_score(self, scene: SceneAnalysis, candidate: dict[str, Any], semantic_score: float):
        metadata_text = candidate.get("metadata_text", "")
        keyword_score = _keyword_overlap_score(scene.keywords, metadata_text)
        base = max(0.0, semantic_score) * 6.0
        base += keyword_score * 2.0

        if scene.strict_entity_required:
            if _candidate_mentions_entity(candidate, scene.main_entity, scene.keywords):
                base += 2.0
            else:
                return 0.0, "Rejected because the candidate metadata does not reference the required entity."

        if scene.time_period and _is_historical_text(scene.time_period):
            if any(term in metadata_text.lower() for term in ["modern", "vlog", "travel video", "cinematic stock"]):
                base -= 2.0

        source = candidate.get("source", "")
        if scene.strict_entity_required and source in {"pexels", "pixabay", "coverr"}:
            base -= 1.5
        if not scene.strict_entity_required and source in {"pexels", "pixabay"}:
            base += 0.5

        return max(0.0, min(10.0, base)), "Heuristic fallback score."

    def _llm_score_candidate(self, scene: SceneAnalysis, candidate: dict[str, Any], semantic_score: float):
        if not self.client:
            return self._heuristic_llm_score(scene, candidate, semantic_score)

        prompt = f"""
You are a strict video relevance evaluator.

Sentence:
{scene.sentence}

Main Entity:
{scene.main_entity or "None"}

Entity Type:
{scene.entity_type}

Strict Entity Required:
{str(scene.strict_entity_required).lower()}

Context:
{scene.context_type}

Time Period:
{scene.time_period or "Unknown"}

Location:
{scene.location or "Unknown"}

Visual Intent:
{scene.visual_intent}

Video Source:
{candidate.get("source", "")}

Video Title:
{candidate.get("title", "")}

Description:
{candidate.get("description", "")}

Tags:
{", ".join(candidate.get("tags", []) or [])}

Rules:
1. Must match the main entity if required.
2. Must match the context and visual intent.
3. Must match time period and location.
4. Reject generic or misleading visuals.

Return JSON only:
{{
  "score": number,
  "reason": "short explanation"
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            parsed = _extract_json_payload(response.choices[0].message.content or "")
            score = float(parsed.get("score", 0))
            reason = _normalize_space(parsed.get("reason", ""))
            return max(0.0, min(10.0, score)), reason or "LLM reranked candidate."
        except Exception:
            return self._heuristic_llm_score(scene, candidate, semantic_score)

    def _validate_candidate(self, scene: SceneAnalysis, candidate: dict[str, Any], score: float):
        if score < 8.0:
            return False, "Score below hard filter threshold."
        if not self.client:
            if scene.strict_entity_required and not _candidate_mentions_entity(candidate, scene.main_entity, scene.keywords):
                return False, "Failed entity validation."
            return True, "Heuristic validation passed."

        prompt = f"""
Is this video a PERFECT match for the sentence?

Sentence: {scene.sentence}
Main Entity: {scene.main_entity or "None"}
Strict Entity Required: {str(scene.strict_entity_required).lower()}
Time Period: {scene.time_period or "Unknown"}
Location: {scene.location or "Unknown"}
Visual Intent: {scene.visual_intent}
Video Title: {candidate.get("title", "")}
Description: {candidate.get("description", "")}
Tags: {", ".join(candidate.get("tags", []) or [])}

Return JSON only:
{{
  "match": true_or_false,
  "reason": "short explanation"
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            parsed = _extract_json_payload(response.choices[0].message.content or "")
            return _coerce_bool(parsed.get("match")), _normalize_space(parsed.get("reason", ""))
        except Exception:
            return True, "Validation fallback accepted candidate."

    def _apply_hard_filters(self, scene: SceneAnalysis, candidates: list[dict[str, Any]]):
        filtered = []
        for candidate in candidates:
            metadata_text = candidate.get("metadata_text", "").lower()
            negative_hit = any(keyword.lower() in metadata_text for keyword in scene.negative_keywords if len(keyword) > 3)
            if negative_hit:
                continue
            if scene.strict_entity_required and not _candidate_mentions_entity(candidate, scene.main_entity, scene.keywords):
                continue
            filtered.append(candidate)
        return filtered

    def _fallback_candidate(self, scene: SceneAnalysis):
        fallback_query = _normalize_space(
            " ".join(
                part
                for part in [
                    scene.visual_intent,
                    scene.location,
                    scene.time_period,
                    "historical footage" if _is_historical_text(scene.time_period) else "documentary footage",
                ]
                if part
            )
        )
        links, source = get_best_visual_source(fallback_query, limit=1)
        if not links:
            return None
        url = links[0]
        return {
            "id": f"fallback:{source}:{abs(hash(url))}",
            "source": source or "fallback",
            "url": url,
            "page_url": url,
            "title": f"Fallback visual for {scene.sentence[:80]}",
            "description": fallback_query,
            "tags": scene.keywords,
            "thumbnail_url": None,
            "duration": None,
            "metadata_text": fallback_query,
        }

    def select_candidate(self, scene: SceneAnalysis):
        queries = self.generate_queries(scene)
        raw_candidates = search_multi_source_candidates(queries, per_query_limit=15, total_limit=20)
        candidates = self._apply_hard_filters(scene, raw_candidates)
        if not candidates and not scene.strict_entity_required:
            candidates = raw_candidates

        ranked = self._semantic_rank(scene, candidates, top_k=15) if candidates else []
        for candidate, semantic_score in ranked:
            score, reason = self._llm_score_candidate(scene, candidate, semantic_score)
            if score < 8.0:
                continue
            passed, validation_reason = self._validate_candidate(scene, candidate, score)
            if passed:
                scene.selected_candidate = candidate
                scene.rerank_reason = reason or validation_reason
                scene.llm_score = score
                scene.validation_passed = True
                return scene

        fallback = self._fallback_candidate(scene)
        if fallback:
            scene.selected_candidate = fallback
            scene.fallback_used = True
            scene.rerank_reason = "No strict match cleared the threshold, so the pipeline fell back to contextual symbolic footage."
            scene.llm_score = 0.0
        return scene


def build_visual_plan(script_text: str, desired_scene_count: int | None = None, model: str = "gpt-4o-mini"):
    intelligence = ScriptIntelligenceEngine(model=model)
    selector = CandidateSelector(model=model)
    context = GlobalContextMemory()

    units = intelligence.segment_text(script_text, preferred_scene_count=desired_scene_count)
    scenes = []
    for unit in units:
        scene = intelligence.analyze_sentence(unit, context)
        selector.select_candidate(scene)
        context.absorb(scene)
        scenes.append(scene.to_dict())
        if desired_scene_count and len(scenes) >= desired_scene_count:
            break
    return scenes
