"""Shared resolver pipeline.

A ``ResolverPipeline`` composes the superset of stages every ``resolve_*``
function performs (preprocess → deduplicate → local lookup →
disambiguate/build → enrich → fallback → cache write → re-expand → report).
Stages are optional; the pipeline supplies no-op defaults, so a cache-only
resolver omits ``fallbacks``/``cache_sink`` rather than branching internally.

See ``specs/resolver-framework.md`` for the design. The first resolver migrated
onto it is ``resolve_guide_sequences`` (see ``auto_atlas.guide_rna``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from auto_atlas.types import Resolution, ResolutionReport

# ---------------------------------------------------------------------------
# Per-run context and intermediate state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolverContext:
    """Per-run context passed to every stage."""

    organism: str | None = None
    tool: str = "resolve_unknown"
    # Extensible bag for entity, min_similarity, assembly, species, etc.
    extras: dict[str, object] = field(default_factory=dict)


@dataclass
class LookupHit:
    """Intermediate match before disambiguation / enrichment."""

    key: str  # normalized lookup key
    candidates: list[dict]  # raw rows or API payloads
    source: str  # e.g. "lancedb", "reference_db_synonym"
    original: str = ""  # caller's first-seen string; backfilled by the pipeline


@dataclass
class Disambiguation:
    """Outcome of picking among a hit's candidates.

    ``chosen`` is the winning *raw candidate* (a row dict or payload from
    ``LookupHit.candidates``), not a flattened string — the result builder reads
    typed fields off it. ``chosen=None`` means no acceptable pick.
    """

    chosen: dict | None
    confidence: float
    source: str
    alternatives: list[str] = field(default_factory=list)


@dataclass
class PipelineState[R: Resolution]:
    """Mutable state threaded through pipeline stages."""

    inputs: list[str]
    context: ResolverContext
    # per-input normalized key (aligned with ``inputs``)
    normalized: list[str] = field(default_factory=list)
    # normalized_key → original (first-seen casing)
    key_map: dict[str, str] = field(default_factory=dict)
    unique_keys: list[str] = field(default_factory=list)
    # normalized_key → LookupHit or None (miss)
    local_hits: dict[str, LookupHit | None] = field(default_factory=dict)
    # normalized_key → fully built resolution
    results: dict[str, R] = field(default_factory=dict)
    # keys resolved from a local hit (used to skip fallback + cache write-back)
    from_local: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Stage protocols
# ---------------------------------------------------------------------------


class Preprocessor(Protocol):
    def __call__(self, value: str, ctx: ResolverContext) -> str: ...


class LocalLookup[R: Resolution](Protocol):
    def lookup(self, keys: list[str], ctx: ResolverContext) -> dict[str, LookupHit | None]: ...


class Disambiguator(Protocol):
    def pick(self, hit: LookupHit, ctx: ResolverContext) -> Disambiguation: ...


class ResultBuilder[R: Resolution](Protocol):
    """Build the typed ``Resolution`` for a key from its disambiguation.

    Owns both paths: ``picked is None`` (a miss that survived every fallback)
    yields the unresolved stub, which still carries resolver-specific context
    fields (``organism`` for genes/proteins, ``ontology_name`` for ontologies).
    """

    def build(
        self, key: str, original: str, picked: Disambiguation | None, ctx: ResolverContext
    ) -> R: ...


class Enricher[R: Resolution](Protocol):
    # Batch over the whole result set: genes/proteins enrich with a single
    # secondary lookup keyed by the resolved id, so a per-item hook would fan
    # out into one query per result. May mutate in place and return the dict.
    def enrich(self, results: dict[str, R], ctx: ResolverContext) -> dict[str, R]: ...


class Fallback[R: Resolution](Protocol):
    def try_resolve(self, key: str, original: str, ctx: ResolverContext) -> R | None: ...


class CacheSink[R: Resolution](Protocol):
    def to_record(self, result: R, ctx: ResolverContext) -> dict | None: ...
    def write(self, records: list[dict]) -> None: ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class ResolverPipeline[R: Resolution]:
    tool: str
    result_builder: ResultBuilder[R]  # builds resolved results and unresolved stubs

    preprocessor: Preprocessor | None = None
    prescan_fallbacks: list[Fallback[R]] = field(default_factory=list)
    local_lookup: LocalLookup[R] | None = None
    disambiguator: Disambiguator | None = None
    enricher: Enricher[R] | None = None
    fallbacks: list[Fallback[R]] = field(default_factory=list)
    cache_sink: CacheSink[R] | None = None

    def resolve(
        self, values: list[str], *, tool: str | None = None, **ctx_kwargs
    ) -> ResolutionReport:
        # ``tool`` overrides the report/provenance label per call, so resolvers
        # that share one implementation under several public names (the ontology
        # wrappers) can each stamp their own tool. Defaults to ``self.tool``.
        ctx = ResolverContext(tool=tool or self.tool, **ctx_kwargs)
        state: PipelineState[R] = PipelineState(inputs=list(values), context=ctx)
        self._run_preprocess(state)
        self._run_deduplicate(state)
        self._run_prescan_fallbacks(state)
        self._run_local_lookup(state)
        self._run_disambiguate_and_build(state)
        self._run_enrich(state)
        self._run_fallbacks(state)
        self._run_cache_write(state)
        return self._reexpand_and_report(state)

    # -- stages -------------------------------------------------------------

    def _run_preprocess(self, state: PipelineState[R]) -> None:
        if self.preprocessor is None:
            state.normalized = list(state.inputs)
        else:
            state.normalized = [self.preprocessor(v, state.context) for v in state.inputs]

    def _run_deduplicate(self, state: PipelineState[R]) -> None:
        for original, key in zip(state.inputs, state.normalized, strict=True):
            if key not in state.key_map:
                state.key_map[key] = original
                state.unique_keys.append(key)

    def _run_prescan_fallbacks(self, state: PipelineState[R]) -> None:
        for fb in self.prescan_fallbacks:
            for key in state.unique_keys:
                if key in state.results:
                    continue
                res = fb.try_resolve(key, state.key_map[key], state.context)
                if res is not None:
                    state.results[key] = res

    def _run_local_lookup(self, state: PipelineState[R]) -> None:
        if self.local_lookup is None:
            return
        pending = [key for key in state.unique_keys if key not in state.results]
        hits = self.local_lookup.lookup(pending, state.context)
        for key, hit in hits.items():
            if hit is not None and not hit.original:
                hit.original = state.key_map[key]
            state.local_hits[key] = hit

    def _run_disambiguate_and_build(self, state: PipelineState[R]) -> None:
        for key in state.unique_keys:
            if key in state.results:  # handled by a prescan short-circuit
                continue
            hit = state.local_hits.get(key)
            if hit is None:
                continue  # genuine miss → leave for fallbacks
            if self.disambiguator is not None:
                picked = self.disambiguator.pick(hit, state.context)
            else:
                chosen = hit.candidates[0] if hit.candidates else None
                picked = Disambiguation(
                    chosen=chosen,
                    confidence=1.0 if chosen is not None else 0.0,
                    source=hit.source,
                )
            state.results[key] = self.result_builder.build(
                key, state.key_map[key], picked, state.context
            )
            state.from_local.add(key)

    def _run_enrich(self, state: PipelineState[R]) -> None:
        if self.enricher is None:
            return
        state.results = self.enricher.enrich(state.results, state.context)

    def _run_fallbacks(self, state: PipelineState[R]) -> None:
        if not self.fallbacks:
            return
        for key in state.unique_keys:
            if key in state.results:  # resolved locally or by a prescan
                continue
            original = state.key_map[key]
            for fb in self.fallbacks:
                res = fb.try_resolve(key, original, state.context)
                if res is not None:
                    state.results[key] = res
                    break

    def _run_cache_write(self, state: PipelineState[R]) -> None:
        if self.cache_sink is None:
            return
        records: list[dict] = []
        for key, result in state.results.items():
            if key in state.from_local:
                continue  # came from cache; do not rewrite
            record = self.cache_sink.to_record(result, state.context)
            if record is not None:
                records.append(record)
        if records:
            self.cache_sink.write(records)

    def _reexpand_and_report(self, state: PipelineState[R]) -> ResolutionReport:
        results: list[R] = []
        for original, key in zip(state.inputs, state.normalized, strict=True):
            result = state.results.get(key)
            if result is None:  # no hit, no fallback — emit the unresolved stub
                result = self.result_builder.build(key, original, None, state.context)
            results.append(result)

        resolved = sum(1 for r in results if r.resolved_value is not None)
        ambiguous = sum(1 for r in results if len(r.alternatives) > 0)
        return ResolutionReport(
            tool=state.context.tool,
            total=len(results),
            resolved=resolved,
            unresolved=len(results) - resolved,
            ambiguous=ambiguous,
            results=results,
        )
