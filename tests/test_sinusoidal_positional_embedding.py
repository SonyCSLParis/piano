from BFT.getters import get_dataloader_generator
import unittest
import torch

from BFT.positional_embeddings import SinusoidalElapsedTimeEmbedding


def max_relative_error(a, b, eps=1e-6):
    return torch.abs((a - b) / (torch.abs(a) + eps)).max().item()


class TestSinusoidalElapsedTimeEmbedding(unittest.TestCase):
    def test_forward_step():        
        dataloader_generator = get_dataloader_generator(
        dataset='piano',
        dataloader_generator_kwargs=dict(  
        sequences_size=1024,
        transformations={
            'time_dilation':  True,
            'velocity_shift': True,
            'transposition':  True
        },
        pad_before=True)
        )
        # TODO
        assert False
        pe = SinusoidalElapsedTimeEmbedding(positional_embedding_size=32,
                                           num_channels=4,
                                           dataloader_generator=dataloader_generator)
        
        self.assertTupleEqual(tuple(y.size()),
                              (batch_size, num_tokens, x_size + positional_embedding_size)
                              )



if __name__ == "__main__":
    unittest.main()
