"""Shared resolver framework (see ``specs/resolver-framework.md``)."""

from auto_atlas.resolvers.common import AliasLookup, CanonicalAliasDisambiguator
from auto_atlas.resolvers.pipeline import (
    CacheSink,
    Disambiguation,
    Disambiguator,
    Enricher,
    Fallback,
    LocalLookup,
    LookupHit,
    PipelineState,
    Preprocessor,
    ResolverContext,
    ResolverPipeline,
    ResultBuilder,
)

__all__ = [
    "AliasLookup",
    "CacheSink",
    "CanonicalAliasDisambiguator",
    "Disambiguation",
    "Disambiguator",
    "Enricher",
    "Fallback",
    "LocalLookup",
    "LookupHit",
    "PipelineState",
    "Preprocessor",
    "ResolverContext",
    "ResolverPipeline",
    "ResultBuilder",
]
