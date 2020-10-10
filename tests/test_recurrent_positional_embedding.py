import unittest
import torch

from BFT.positional_embeddings.recurrent_positional_embedding import RecurrentPositionalEmbedding


def max_relative_error(a, b, eps=1e-6):
    return torch.abs((a - b) / (torch.abs(a) + eps)).max().item()


class TestRecurrentPositionalEmbedding(unittest.TestCase):
    def test_correct_dimensions(self):
        batch_size = 8
        positional_embedding_size = 8
        num_channels = 4
        num_tokens = num_channels * 10
        x_size = 36

        pe = RecurrentPositionalEmbedding(
            positional_embedding_size=positional_embedding_size,
            num_channels=num_channels
        )

        x = torch.randn((batch_size, num_tokens, x_size))

        y, h = pe(x)
        self.assertTupleEqual(tuple(y.size()),
                              (batch_size, num_tokens, x_size + positional_embedding_size)
                              )

    def test_correct_dimensions_cuda(self):
        batch_size = 8
        positional_embedding_size = 8
        num_channels = 4
        num_tokens = num_channels * 10
        x_size = 36

        pe = RecurrentPositionalEmbedding(
            positional_embedding_size=positional_embedding_size,
            num_channels=num_channels
        ).cuda()

        x = torch.randn((batch_size, num_tokens, x_size)).cuda()

        y, h = pe(x)
        self.assertTupleEqual(tuple(y.size()),
                              (batch_size, num_tokens, x_size + positional_embedding_size)
                              )

    def test_equality_recurrent_non_recurrent(self):
        batch_size = 8
        positional_embedding_size = 8
        num_channels = 4
        num_tokens = num_channels * 10
        x_size = 36

        pe = RecurrentPositionalEmbedding(
            positional_embedding_size=positional_embedding_size,
            num_channels=num_channels
        )

        # deactivate dropout:
        pe.eval()

        # first method:
        x = torch.randn((batch_size, num_tokens, x_size))
        y1, h1 = pe(x)

        # second method
        h2 = None
        y2 = torch.zeros_like(y1)
        for i in range(num_tokens):
            input = x[:, i:i + 1]
            output, h2 = pe(input, i=i, h=h2)
            y2[:, i:i + 1] = output

        self.assertLess(max_relative_error(y1, y2), 1e-5)
        self.assertLess(max_relative_error(h1, h2), 1e-5)


if __name__ == "__main__":
    unittest.main()
