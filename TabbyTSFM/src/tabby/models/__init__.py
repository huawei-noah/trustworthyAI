# Copyright contributors to the TSFM project
#

from typing import TYPE_CHECKING

# rely on isort to merge the imports
from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_patchtst_fm": [
        "PATCHTSTFM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PatchTSTFMConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_patchtst_fm"] = [
        "PatchTSTFMPreTrainedModel",
        "PatchTSTFMModel",
        "PatchTSTFMForPretraining",
        "PatchTSTFMForPrediction",
    ]


if TYPE_CHECKING:
    from .configuration_patchtst_fm import (
        PATCHTSTFM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PatchTSTFMConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_patchtst_fm import (
            PatchTSTFMForPretraining,
            PatchTSTFMForPrediction,
            PatchTSTFMPreTrainedModel,
            PatchTSTFMModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
