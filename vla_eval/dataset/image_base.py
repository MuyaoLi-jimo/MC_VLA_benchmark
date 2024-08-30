from vla_eval.dataset.base import BaseDataset

class ImageBaseDataset(BaseDataset):
    MODALITY = 'IMAGE'
    def __init__(self, dataset):
        super().__init__(dataset)
        
    