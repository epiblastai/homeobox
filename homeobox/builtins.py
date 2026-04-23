"""Built-in feature space specs and their reconstructors.

Imported at package init time to register all built-in specs before
user code defines schema subclasses or runs queries.
"""

import numpy as np

from homeobox.fragments.reconstruction import IntervalReconstructor
from homeobox.group_specs import (
    ArraySpec,
    LayersSpec,
    PointerKind,
    ZarrGroupSpec,
    register_spec,
)
from homeobox.reconstruction import DenseReconstructor, SparseCSRReconstructor

# ---------------------------------------------------------------------------
# Built-in specs
# ---------------------------------------------------------------------------

GENE_EXPRESSION_SPEC = ZarrGroupSpec(
    feature_space="gene_expression",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="csr/indices", ndim=1, allowed_dtypes=[np.uint32]),
    ],
    layers=LayersSpec(
        prefix="csr",
        match_shape_of="csr/indices",
        required=[ArraySpec(array_name="counts", ndim=1, allowed_dtypes=[np.uint32])],
        allowed=[
            ArraySpec(array_name="counts", ndim=1, allowed_dtypes=[np.uint32]),
            ArraySpec(array_name="log_normalized", ndim=1, allowed_dtypes=[np.float32]),
            ArraySpec(array_name="tpm", ndim=1, allowed_dtypes=[np.float32]),
        ],
    ),
    reconstructor=SparseCSRReconstructor(),
)

IMAGE_FEATURES_SPEC = ZarrGroupSpec(
    feature_space="image_features",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
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
    reconstructor=DenseReconstructor(),
)

# ---------------------------------------------------------------------------
# Protein abundance (CITE-seq / ADT)
# ---------------------------------------------------------------------------

PROTEIN_ABUNDANCE_SPEC = ZarrGroupSpec(
    feature_space="protein_abundance",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    layers=LayersSpec(
        required=[ArraySpec(array_name="counts", ndim=2, allowed_dtypes=[np.uint32])],
        allowed=[
            ArraySpec(array_name="counts", ndim=2, allowed_dtypes=[np.uint32]),
            ArraySpec(array_name="clr_normalized", ndim=2, allowed_dtypes=[np.float32]),
            ArraySpec(array_name="dsb_normalized", ndim=2, allowed_dtypes=[np.float32]),
        ],
    ),
    reconstructor=DenseReconstructor(),
)

# ---------------------------------------------------------------------------
# Chromatin accessibility (cell-sorted fragments)
# ---------------------------------------------------------------------------

CHROMATIN_ACCESSIBILITY_SPEC = ZarrGroupSpec(
    feature_space="chromatin_accessibility",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="cell_sorted/chromosomes", ndim=1, allowed_dtypes=[np.uint8]),
        ArraySpec(array_name="cell_sorted/starts", ndim=1, allowed_dtypes=[np.uint32]),
        ArraySpec(array_name="cell_sorted/lengths", ndim=1, allowed_dtypes=[np.uint16, np.uint32]),
    ],
    layers=LayersSpec(),
    reconstructor=IntervalReconstructor(),
)

# ---------------------------------------------------------------------------
# Image tiles
# ---------------------------------------------------------------------------

IMAGE_TILES_SPEC = ZarrGroupSpec(
    feature_space="image_tiles",
    pointer_kind=PointerKind.DENSE,
    has_var_df=False,
    required_arrays=[
        ArraySpec(array_name="data", ndim=4, allowed_dtypes=[np.uint8, np.uint16]),
    ],
    reconstructor=DenseReconstructor(),
)


for _spec in [
    GENE_EXPRESSION_SPEC,
    IMAGE_FEATURES_SPEC,
    PROTEIN_ABUNDANCE_SPEC,
    CHROMATIN_ACCESSIBILITY_SPEC,
    IMAGE_TILES_SPEC,
]:
    register_spec(_spec)
