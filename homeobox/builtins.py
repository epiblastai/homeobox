"""Built-in feature space specs and their reconstructors.

Imported at package init time to register all built-in specs before
user code defines schema subclasses or runs queries.
"""

from homeobox.fragments.reconstruction import IntervalReconstructor
from homeobox.group_specs import (
    ArraySpec,
    DTypeKind,
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
        ArraySpec(array_name="csr/indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    layers=LayersSpec(
        prefix="csr",
        uniform_shape=True,
        match_shape_of="csr/indices",
        required=["counts"],
        allowed=["counts", "log_normalized", "tpm"],
    ),
    reconstructor=SparseCSRReconstructor(),
)

IMAGE_FEATURES_SPEC = ZarrGroupSpec(
    feature_space="image_features",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    layers=LayersSpec(
        uniform_shape=True,
        required=["ctrl_standardized"],
        allowed=["raw", "log_normalized", "ctrl_standardized"],
    ),
    reconstructor=DenseReconstructor(),
)

PROTEIN_ABUNDANCE_SPEC = ZarrGroupSpec(
    feature_space="protein_abundance",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    layers=LayersSpec(
        uniform_shape=True,
        required=["counts"],
        allowed=["counts", "clr_normalized", "dsb_normalized"],
    ),
    reconstructor=DenseReconstructor(),
)


CHROMATIN_ACCESSIBILITY_SPEC = ZarrGroupSpec(
    feature_space="chromatin_accessibility",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(
            array_name="cell_sorted/chromosomes", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER
        ),
        ArraySpec(array_name="cell_sorted/starts", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ArraySpec(array_name="cell_sorted/lengths", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    layers=LayersSpec(),
    reconstructor=IntervalReconstructor(),
)


IMAGE_TILES_SPEC = ZarrGroupSpec(
    feature_space="image_tiles",
    pointer_kind=PointerKind.DENSE,
    has_var_df=False,
    required_arrays=[
        ArraySpec(array_name="data", ndim=4, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    reconstructor=DenseReconstructor(),
)


EMBEDDING_SPEC = ZarrGroupSpec(
    feature_space="embedding",
    pointer_kind=PointerKind.DENSE,
    # The var_df for embeddings is trivial dimensionality
    has_var_df=False,
    layers=LayersSpec(
        uniform_shape=True,
        required=["raw"],
        allowed=["ctrl_standardized", "pca_whitened"],
    ),
    reconstructor=DenseReconstructor(),
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
