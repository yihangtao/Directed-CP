# # -*- coding: utf-8 -*-
# # Author: Runsheng Xu <rxx3386@ucla.edu>
# # License: TDG-Attribution-NonCommercial-NoDistrib

# from opencood.data_utils.datasets.early_fusion_dataset_dair import EarlyFusionDatasetDAIR
# from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
# from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
# from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR

# __all__ = {
#     'EarlyFusionDatasetDAIR': EarlyFusionDatasetDAIR,
#     'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
#     'IntermediateFusionDataset': IntermediateFusionDataset,
#     'LateFusionDatasetDAIR': LateFusionDatasetDAIR
# }

# # the final range for evaluation
# GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
# GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
# # The communication range for cavs
# COM_RANGE = 70


# def build_dataset(dataset_cfg, visualize=False, train=True):
#     dataset_name = dataset_cfg['fusion']['core_method']
#     error_message = f"{dataset_name} is not found. " \
#                     f"Please add your processor file's name in opencood/" \
#                     f"data_utils/datasets/init.py"

#     dataset = __all__[dataset_name](
#         params=dataset_cfg,
#         visualize=visualize,
#         train=train
#     )

#     return dataset


from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from opencood.data_utils.datasets.v2xsim_basedataset import V2XSIMBaseDataset
from opencood.data_utils.datasets.dairv2x_basedataset import DAIRV2XBaseDataset


def build_dataset(dataset_cfg, visualize=False, train=True):
    fusion_name = dataset_cfg['fusion']['core_method']
    dataset_name = dataset_cfg['fusion']['dataset']

    assert fusion_name in ['late', 'lateheter', 'intermediate', 'intermediate2stage', 'intermediateheter', 'early', 'intermediateheterinfer']
    assert dataset_name in ['opv2v', 'v2xsim', 'dairv2x', 'v2xset']

    fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
    fusion_dataset_func = eval(fusion_dataset_func)
    base_dataset_cls = dataset_name.upper() + "BaseDataset"
    base_dataset_cls = eval(base_dataset_cls)

    dataset = fusion_dataset_func(base_dataset_cls)(
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
