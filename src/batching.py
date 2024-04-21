import math
import random
from torch.utils.data import Sampler, RandomSampler, SequentialSampler


class TokenSampler(Sampler):
    """
    A Sampler which groups sequences into buckets based on length and constructs batches using
    a (potentially) different number of sequences from each bucket to achieve a target number of
    tokens in each batch. This approach has a number of advantages:
        - Faster training and eval since there are fewer pad tokens vs random batching
        - Potentially improved training stability since the number of tokens is approx the same
          each batch

    Note: There is a systematic error in the batch size (it will be slightly larger than the
          target size on average) since we simply take the mean of the seq lengths in the bucket,
          this does not account for padding that will result from the largest seq in the batch.
    """
    NUM_BUCKETS = 4

    def __init__(
            self,
            seq_lengths,
            tokens_in_batch,
            shuffle=True,
            drop_last=False,
    ):
        """ Init method

        Args:
            num_buckets (int): Number of buckets to split sequences into
            seq_lengths (List[int]): The length of the sequences in the dataset (in the same order)
            batch_size (int): Target number of tokens in each batch
            shuffle (Optional[bool]): Shuffle the indices within each bucket
            drop_last (Optional[bool]): Forget about the indices remaining at the end of each bucket
        """
        super().__init__()
        self.tokens_in_batch = tokens_in_batch
        buckets = [[] for _ in range(self.NUM_BUCKETS)]
        lengths = [[] for _ in range(self.NUM_BUCKETS)]

        min_length = min(seq_lengths)
        max_length = max(seq_lengths) + 1
        bucket_width = (max_length - min_length) / self.NUM_BUCKETS

        bucket_limits = []
        lower_limit = float(min_length)

        # Setup lower (inclusive) and upper (exclusive) seq length limits on buckets
        for _ in range(self.NUM_BUCKETS):
            upper_limit = lower_limit + bucket_width
            bucket_limits.append((lower_limit, upper_limit))
            lower_limit = upper_limit

        # Add indices to correct bucket based on seq length
        for seq_idx, length in enumerate(seq_lengths):
            for b_idx, (lower, upper) in enumerate(bucket_limits):
                if lower <= length < upper:
                    buckets[b_idx].append(seq_idx)
                    lengths[b_idx].append(length)

        if shuffle:
            samplers = [RandomSampler(idxs) for idxs in buckets]
        else:
            samplers = [SequentialSampler(idxs) for idxs in buckets]

        # Work out approx number of sequences required for each bucket
        self.avg_lengths = [sum(ls) // len(ls) for ls in lengths]
        self.rem_seqs = [len(b) for b in lengths]

        self.num_batches = [math.ceil(sum(ls) / tokens_in_batch) for ls in lengths]

        self.buckets = buckets
        self.samplers = samplers

    def __iter__(self):
        iters = [iter(sampler) for sampler in self.samplers]
        rem_seqs = self.rem_seqs[:]
        while sum(rem_seqs) > 0:
            # Choose a bucket
            b_idx = random.choices(range(self.NUM_BUCKETS), weights=rem_seqs, k=1)[0]

            # Decide the number of sequences to take from the bucket
            n_to_take = min(rem_seqs[b_idx], self.tokens_in_batch // self.avg_lengths[b_idx])

            # Take sequences from the bucket
            seq_ids_in_bucket = [next(iters[b_idx]) for _ in range(n_to_take)]
            sequences_in_batch = [self.buckets[b_idx][idx] for idx in seq_ids_in_bucket]
            rem_seqs[b_idx] -= n_to_take
            yield sequences_in_batch

    def __len__(self):
        # ??? It's sometimes off by +1 or -1 compared to the real number of yielded batches.
        # Is it a problem?
        return sum(self.num_batches)
