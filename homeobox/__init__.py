import homeobox.builtins  # noqa: F401  # register built-in specs
import homeobox.codecs.bitpacking  # noqa: F401  # register bitpacking codec
from homeobox.atlas import RaggedAtlas, create_or_open_atlas
from homeobox.dataloader import (
    DenseFeatureBatch,
    MultimodalBatch,
    MultimodalHoxDataset,
    SparseBatch,
    SpatialTileBatch,
    UnimodalHoxDataset,
    make_loader,
)
from homeobox.fragments.genome_query import GenomeSortedReader, RegionResult
from homeobox.fragments.peak_matrix import FragmentCounter, GenomicRange
from homeobox.fragments.reconstruction import FragmentResult, IntervalReconstructor
from homeobox.ingestion import add_anndata_batch, add_csc, add_from_anndata
from homeobox.multimodal import MultimodalResult
from homeobox.pointer_types import (
    DenseZarrPointer,
    SparseZarrPointer,
)
from homeobox.query import AtlasQuery
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    StableUIDBaseSchema,
    StableUIDField,
)

__all__ = [
    "RaggedAtlas",
    "create_or_open_atlas",
    "AtlasQuery",
    "UnimodalHoxDataset",
    "MultimodalHoxDataset",
    "SparseBatch",
    "DenseFeatureBatch",
    "SpatialTileBatch",
    "MultimodalBatch",
    "make_loader",
    "add_from_anndata",
    "add_anndata_batch",
    "add_csc",
    "HoxBaseSchema",
    "FeatureBaseSchema",
    "DatasetSchema",
    "PointerField",
    "StableUIDBaseSchema",
    "StableUIDField",
    "SparseZarrPointer",
    "DenseZarrPointer",
    "FragmentResult",
    "IntervalReconstructor",
    "FragmentCounter",
    "GenomicRange",
    "GenomeSortedReader",
    "RegionResult",
    "MultimodalResult",
]
