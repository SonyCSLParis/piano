from abc import ABCMeta, abstractmethod


class DataloaderGenerator(metaclass=ABCMeta):
    """
    Base abstract class for data loader generators
    dataloaders
    """
    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def dataloaders(self, batch_size, num_workers, shuffle_train=True,
                    shuffle_val=False):
        return

