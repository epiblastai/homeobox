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
    PointerKind,
    ZarrGroupSpec,
    register_spec,
)
from homeobox.reconstruction import DenseReconstructor, SparseGeneExpressionReconstructor

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
    pointer_kind=PointerKind.SPARSE,
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
    pointer_kind=PointerKind.DENSE,
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
    pointer_kind=PointerKind.DENSE,
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


CHROMATIN_ACCESSIBILITY_SPEC = FeatureSpaceSpec(
    feature_space="chromatin_accessibility",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    reconstructor=IntervalReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
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
    ),
)


IMAGE_TILES_SPEC = FeatureSpaceSpec(
    feature_space="image_tiles",
    pointer_kind=PointerKind.DENSE,
    has_var_df=False,
    reconstructor=DenseReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        required_arrays=[
            ArraySpec(array_name="data", ndim=4, allowed_dtypes=[np.float32, np.uint8, np.uint16]),
        ],
    ),
)


EMBEDDING_SPEC = FeatureSpaceSpec(
    feature_space="embedding",
    pointer_kind=PointerKind.DENSE,
    # The var_df for embeddings is trivial dimensionality
    has_var_df=False,
    reconstructor=DenseReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        layers=LayersSpec(
            required=[
                ArraySpec(array_name="raw", ndim=2, allowed_dtypes=[np.float32, np.float16]),
            ],
            allowed=[
                ArraySpec(array_name="raw", ndim=2, allowed_dtypes=[np.float32, np.float16]),
                ArraySpec(
                    array_name="ctrl_standardized",
                    ndim=2,
                    allowed_dtypes=[np.float32, np.float16],
                ),
                ArraySpec(
                    array_name="pca_whitened",
                    ndim=2,
                    allowed_dtypes=[np.float32, np.float16],
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
    EMBEDDING_SPEC,
]:
    register_spec(_spec)
