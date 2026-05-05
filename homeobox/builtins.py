"""Built-in feature space specs and their reconstructors.

Imported at package init time to register all built-in specs before
user code defines schema subclasses or runs queries.
"""

import numpy as np

from homeobox.codecs.bitpacking import BitpackingCodec
from homeobox.fragments.reconstruction import IntervalReconstructor
from homeobox.group_specs import (
    ArraySpec,
    FeatureSpaceSpec,
    LayersSpec,
    ZarrGroupSpec,
    register_spec,
)
from homeobox.reconstruction import DenseReconstructor, SparseGeneExpressionReconstructor
from homeobox.schema import DenseZarrPointer, SparseZarrPointer

# ---------------------------------------------------------------------------
# Gene expression (CSR primary, optional CSC feature-oriented copy)
# ---------------------------------------------------------------------------

_GENE_EXPRESSION_LAYER_SPECS = [
    ArraySpec(
        array_name="counts",
        ndim=1,
        allowed_dtypes=[np.uint32],
        compressors=BitpackingCodec(transform="none"),
    ),
    ArraySpec(array_name="log_normalized", ndim=1, allowed_dtypes=[np.float32]),
    ArraySpec(array_name="tpm", ndim=1, allowed_dtypes=[np.float32]),
]

GENE_EXPRESSION_CSR = ZarrGroupSpec(
    required_arrays=[
        ArraySpec(
            array_name="csr/indices",
            ndim=1,
            allowed_dtypes=[np.uint32],
            compressors=BitpackingCodec(transform="delta"),
        ),
    ],
    layers=LayersSpec(
        prefix="csr",
        match_shape_of="csr/indices",
        required=[
            ArraySpec(
                array_name="counts",
                ndim=1,
                allowed_dtypes=[np.uint32],
                compressors=BitpackingCodec(transform="none"),
            ),
        ],
        allowed=_GENE_EXPRESSION_LAYER_SPECS,
    ),
)

GENE_EXPRESSION_CSC = ZarrGroupSpec(
    required_arrays=[
        ArraySpec(
            array_name="csc/indices",
            ndim=1,
            allowed_dtypes=[np.uint32],
        ),
        ArraySpec(
            array_name="csc/indptr",
            ndim=1,
            allowed_dtypes=[np.int64],
        ),
    ],
    layers=LayersSpec(
        prefix="csc",
        match_shape_of="csc/indices",
        required=[
            ArraySpec(array_name="counts", ndim=1, allowed_dtypes=[np.uint32]),
        ],
        allowed=_GENE_EXPRESSION_LAYER_SPECS,
    ),
)

GENE_EXPRESSION_SPEC = FeatureSpaceSpec(
    feature_space="gene_expression",
    pointer_type=SparseZarrPointer,
    has_var_df=True,
    reconstructor=SparseGeneExpressionReconstructor(),
    zarr_group_spec=GENE_EXPRESSION_CSR,
    feature_oriented=GENE_EXPRESSION_CSC,
)

# ---------------------------------------------------------------------------
# Image features (dense)
# ---------------------------------------------------------------------------

IMAGE_FEATURES_SPEC = FeatureSpaceSpec(
    feature_space="image_features",
    pointer_type=DenseZarrPointer,
    has_var_df=True,
    reconstructor=DenseReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        layers=LayersSpec(
            required=[
                ArraySpec(array_name="ctrl_standardized", ndim=2, allowed_dtypes=[np.float32]),
            ],
            allowed=[
                ArraySpec(array_name="raw", ndim=2, allowed_dtypes=[np.float32]),
                ArraySpec(array_name="log_normalized", ndim=2, allowed_dtypes=[np.float32]),
                ArraySpec(array_name="ctrl_standardized", ndim=2, allowed_dtypes=[np.float32]),
            ],
        ),
    ),
)

# ---------------------------------------------------------------------------
# Protein abundance (CITE-seq / ADT)
# ---------------------------------------------------------------------------

PROTEIN_ABUNDANCE_SPEC = FeatureSpaceSpec(
    feature_space="protein_abundance",
    pointer_type=DenseZarrPointer,
    has_var_df=True,
    reconstructor=DenseReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        layers=LayersSpec(
            required=[ArraySpec(array_name="counts", ndim=2, allowed_dtypes=[np.uint32])],
            allowed=[
                ArraySpec(array_name="counts", ndim=2, allowed_dtypes=[np.uint32]),
                ArraySpec(array_name="clr_normalized", ndim=2, allowed_dtypes=[np.float32]),
                ArraySpec(array_name="dsb_normalized", ndim=2, allowed_dtypes=[np.float32]),
            ],
        ),
    ),
)

# ---------------------------------------------------------------------------
# Chromatin accessibility (cell-sorted fragments)
#
# Both layouts are purely structural: the arrays describe fragment
# intervals (chromosome, start, length) and are not feature values.
# A "counts" layer is intentionally omitted — at single-cell resolution
# the signal is sparse enough that per-fragment counts would effectively
# be boolean. Bulk data with real per-fragment counts would add a
# ``counts`` layer here.
# ---------------------------------------------------------------------------

CHROMATIN_ACCESSIBILITY_CELL_SORTED = ZarrGroupSpec(
    required_arrays=[
        ArraySpec(array_name="cell_sorted/chromosomes", ndim=1, allowed_dtypes=[np.uint8]),
        ArraySpec(
            array_name="cell_sorted/starts",
            ndim=1,
            allowed_dtypes=[np.uint32],
            compressors=BitpackingCodec(transform="delta"),
        ),
        ArraySpec(
            array_name="cell_sorted/lengths",
            ndim=1,
            allowed_dtypes=[np.uint16, np.uint32],
            compressors=BitpackingCodec(transform="none"),
        ),
    ],
    layers=LayersSpec(),
)

CHROMATIN_ACCESSIBILITY_GENOME_SORTED = ZarrGroupSpec(
    required_arrays=[
        ArraySpec(
            array_name="genome_sorted/cell_ids",
            ndim=1,
            allowed_dtypes=[np.uint32],
            compressors=BitpackingCodec(transform="none"),
        ),
        ArraySpec(
            array_name="genome_sorted/starts",
            ndim=1,
            allowed_dtypes=[np.uint32],
            compressors=BitpackingCodec(transform="delta"),
        ),
        ArraySpec(
            array_name="genome_sorted/lengths",
            ndim=1,
            allowed_dtypes=[np.uint16, np.uint32],
            compressors=BitpackingCodec(transform="none"),
        ),
        ArraySpec(
            array_name="genome_sorted/chrom_offsets",
            ndim=1,
            allowed_dtypes=[np.int64],
        ),
        ArraySpec(
            array_name="genome_sorted/end_max",
            ndim=1,
            allowed_dtypes=[np.uint32],
        ),
    ],
    layers=LayersSpec(),
)

CHROMATIN_ACCESSIBILITY_SPEC = FeatureSpaceSpec(
    feature_space="chromatin_accessibility",
    pointer_type=SparseZarrPointer,
    has_var_df=True,
    reconstructor=IntervalReconstructor(),
    zarr_group_spec=CHROMATIN_ACCESSIBILITY_CELL_SORTED,
    feature_oriented=CHROMATIN_ACCESSIBILITY_GENOME_SORTED,
)

# ---------------------------------------------------------------------------
# Image tiles
# ---------------------------------------------------------------------------

IMAGE_TILES_SPEC = FeatureSpaceSpec(
    feature_space="image_tiles",
    pointer_type=DenseZarrPointer,
    has_var_df=False,
    reconstructor=DenseReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        layers=LayersSpec(
            required=[
                ArraySpec(
                    array_name="raw", ndim=4, allowed_dtypes=[np.float32, np.uint8, np.uint16]
                ),
            ],
            allowed=[
                ArraySpec(
                    array_name="raw", ndim=4, allowed_dtypes=[np.float32, np.uint8, np.uint16]
                ),
            ],
        ),
    ),
)


for _spec in [
    GENE_EXPRESSION_SPEC,
    IMAGE_FEATURES_SPEC,
    PROTEIN_ABUNDANCE_SPEC,
    CHROMATIN_ACCESSIBILITY_SPEC,
    IMAGE_TILES_SPEC,
]:
    register_spec(_spec)
