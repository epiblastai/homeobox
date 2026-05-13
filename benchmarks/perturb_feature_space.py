"""Bench-local feature space: dense HVG gene expression.

Registers ``hvg_gene_expression`` — a single-layer, 2D, float32 dense feature
space — at module import time. Used by the group-sampler benchmark synth and
the adapters so the bench doesn't piggyback on ``image_features``.

Import for side effects:

    import perturb_feature_space  # noqa: F401

Both ``make_perturbation_synth.py`` and ``benchmark_group_sampler.py`` import
this module before opening or creating an atlas with the
``PerturbCellSchema``.
"""

from __future__ import annotations

import numpy as np

from homeobox.group_specs import (
    ArraySpec,
    FeatureSpaceSpec,
    LayersSpec,
    ZarrGroupSpec,
    register_spec,
    registered_feature_spaces,
)
from homeobox.pointer_types import DenseZarrPointer
from homeobox.reconstruction import DenseFeatureReconstructor

FEATURE_SPACE = "hvg_gene_expression"
LAYER = "expression"

HVG_GENE_EXPRESSION_SPEC = FeatureSpaceSpec(
    feature_space=FEATURE_SPACE,
    pointer_type=DenseZarrPointer,
    has_var_df=True,
    reconstructor=DenseFeatureReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        layers=LayersSpec(
            required=[
                ArraySpec(array_name=LAYER, ndim=2, allowed_dtypes=[np.float32]),
            ],
            allowed=[
                ArraySpec(array_name=LAYER, ndim=2, allowed_dtypes=[np.float32]),
            ],
        ),
    ),
)

if FEATURE_SPACE not in registered_feature_spaces():
    register_spec(HVG_GENE_EXPRESSION_SPEC)
