import torch
from typing import Sequence, Callable, Tuple, Any, Union


SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: SequenceOrTensor,
                 target: SequenceOrTensor,
                 transform: Callable = None,
                 target_transform: Callable = None) -> None:
        super().__init__()
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        datum, target = self.data[idx], self.target[idx]
        if self.transform is not None:
            datum = self.transform(datum)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return datum, target
