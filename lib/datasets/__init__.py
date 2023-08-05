from lib.datasets.synapse import SynapseDataset
from lib.datasets.cityscapes import CityScapesDataset


def get_dataset_by_name(dataset_name):
    available_datasets = {
        "synapse": SynapseDataset,
        "cityscapes": CityScapesDataset,
    }

    if dataset_name.lower() not in available_datasets:
        raise ValueError(f"Invalid dataset name. Available datasets are: {list(available_datasets.keys())}")
    
    dataset = available_datasets[dataset_name.lower()]
    return dataset