from abc import ABCMeta, abstractmethod
from typing import Iterable
from torch import nn
import torch

class BaseSOSEmbedding(nn.Module, metaclass=ABCMeta):
    """Abstract class for handling Start of Sequence embeddings

    Meant to be used only in SOSEmbedding class
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, metadata_dict={}):
        """
        Returns (batch_size, embedding_dim) tensor
        """
        # compute SOS from metadata_dict
        return

class SOSEmbedding(nn.Module):
    """Start of Sequence embedding constructed from a list of
    BaseSOSEmbedding
    """
    def __init__(self, base_sos_embedding_list: Iterable[BaseSOSEmbedding]
                 ) -> None:
        super(SOSEmbedding, self).__init__()
        self.base_sos_embeddings = nn.ModuleList(
            base_sos_embedding_list
        )

    def forward(self, metadata_dict={}):
        sos_list = []
        for embedding in self.base_sos_embeddings:
            sos_list.append(
                embedding(metadata_dict)
            )
        sos = torch.cat(sos_list, dim=1)
        return sos
