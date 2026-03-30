import homeobox.builtins  # noqa: F401  # register built-in specs
import homeobox.codecs.bitpacking  # noqa: F401  # register bitpacking codec

__all__ = [
    # Atlas
    "RaggedAtlas",
    # Query
    "AtlasQuery",
    # Dataloader
    "CellDataset",
    "MultimodalCellDataset",
    "SparseBatch",
    "DenseBatch",
    "MultimodalBatch",
    "make_loader",
    "sparse_to_dense_collate",
    # Samplers
    "CellSampler",
    # Ingestion
    "add_from_anndata",
    "add_anndata_batch",
    "add_csc",
    # Schema
    "HoxBaseSchema",
    "FeatureBaseSchema",
    "DatasetRecord",
    # Fragments
    "FragmentResult",
    "IntervalReconstructor",
    "FragmentCounter",
    "GenomicRange",
    "GenomeSortedReader",
    "RegionResult",
    # Multimodal
    "MultimodalResult",
]


def __getattr__(name: str):
    """Lazy imports for public API to avoid heavy import costs at init."""
    _import_map = {
        "RaggedAtlas": "homeobox.atlas",
        "AtlasQuery": "homeobox.query",
        "CellDataset": "homeobox.dataloader",
        "MultimodalCellDataset": "homeobox.dataloader",
        "SparseBatch": "homeobox.dataloader",
        "DenseBatch": "homeobox.dataloader",
        "MultimodalBatch": "homeobox.dataloader",
        "make_loader": "homeobox.dataloader",
        "sparse_to_dense_collate": "homeobox.dataloader",
        "CellSampler": "homeobox.sampler",
        "add_from_anndata": "homeobox.ingestion",
        "add_anndata_batch": "homeobox.ingestion",
        "add_csc": "homeobox.ingestion",
        "HoxBaseSchema": "homeobox.schema",
        "FeatureBaseSchema": "homeobox.schema",
        "DatasetRecord": "homeobox.schema",
        "FragmentResult": "homeobox.fragments.reconstruction",
        "IntervalReconstructor": "homeobox.fragments.reconstruction",
        "FragmentCounter": "homeobox.fragments.peak_matrix",
        "GenomicRange": "homeobox.fragments.peak_matrix",
        "GenomeSortedReader": "homeobox.fragments.genome_query",
        "RegionResult": "homeobox.fragments.genome_query",
        "MultimodalResult": "homeobox.multimodal",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module 'homeobox' has no attribute {name!r}")
