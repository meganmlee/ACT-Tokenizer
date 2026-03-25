"""
Action tokenizer wrappers for ACT.

This module provides a common interface (ActionTokenizer) for discrete action
tokenization and a FAST+ implementation (FASTTokenizerWrapper). Any new
tokenizer (VQ-VAE, mu-law bins, etc.) should subclass ActionTokenizer.

Usage:
    # Train and save a FAST tokenizer
    python tokenizer.py --dataset_path /path/to/libero_90 --save_path ./fast_tokenizer

    # Or use programmatically:
    wrapper = FASTTokenizerWrapper.train_on_libero(dataset_path, chunk_size=50)
    tokens, token_lens = wrapper.encode(action_chunks)   # (B, max_tokens) LongTensor
    actions = wrapper.decode(tokens, token_lens)          # (B, chunk_size, action_dim) float

    # Load any saved tokenizer by type:
    wrapper = load_tokenizer('./fast_tokenizer')
"""

import numpy as np
import torch
import os
import glob
import h5py
from abc import ABC, abstractmethod


class ActionTokenizer(ABC):
    """
    Abstract interface for action tokenizers used with ACT.

    Any tokenizer must expose these properties and methods so that the rest
    of the codebase (DETRVAE, ACTPolicy, datasets, eval loop) can treat it
    as a drop-in black box.

    Properties (set in __init__):
        vocab_size:     int — number of valid token IDs (for nn.Embedding size = vocab_size + 1)
        max_token_len:  int — padded sequence length
        pad_token_id:   int — token ID used for padding (must be >= vocab_size)
        action_dim:     int — dimensionality of continuous actions
        chunk_size:     int — number of timesteps per action chunk
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def max_token_len(self) -> int: ...

    @property
    @abstractmethod
    def pad_token_id(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def chunk_size(self) -> int: ...

    @abstractmethod
    def encode(self, action_chunks):
        """
        Encode continuous action chunks to padded token sequences.

        Args:
            action_chunks: numpy array (B, chunk_size, action_dim) or (chunk_size, action_dim)

        Returns:
            tokens:     LongTensor (B, max_token_len) — padded with pad_token_id
            token_lens: LongTensor (B,) — actual length of each token sequence
        """
        ...

    @abstractmethod
    def decode(self, tokens, token_lens=None):
        """
        Decode token sequences back to continuous action chunks.

        Args:
            tokens:     LongTensor (B, max_token_len) or (max_token_len,)
            token_lens: LongTensor (B,) or scalar — actual token lengths (optional)

        Returns:
            actions: numpy array (B, chunk_size, action_dim)
        """
        ...

    @abstractmethod
    def save(self, path):
        """Save tokenizer to disk. Must write a 'tokenizer_type' file."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load tokenizer from disk."""
        ...

    def _write_type_marker(self, path):
        """Write the tokenizer type marker so load_tokenizer() can dispatch."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer_type'), 'w') as f:
            f.write(type(self).__name__)


# Registry of tokenizer classes — add new ones here
_TOKENIZER_REGISTRY = {}


def register_tokenizer(cls):
    """Decorator to register a tokenizer class for auto-loading."""
    _TOKENIZER_REGISTRY[cls.__name__] = cls
    return cls


def load_tokenizer(path):
    """
    Load any saved tokenizer by reading its type marker.

    Args:
        path: directory containing a saved tokenizer

    Returns:
        ActionTokenizer subclass instance
    """
    type_path = os.path.join(path, 'tokenizer_type')
    if not os.path.exists(type_path):
        # Backwards compat: assume FAST if no marker
        return FASTTokenizerWrapper.load(path)
    with open(type_path, 'r') as f:
        cls_name = f.read().strip()
    if cls_name not in _TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer type '{cls_name}'. "
                         f"Registered: {list(_TOKENIZER_REGISTRY.keys())}")
    return _TOKENIZER_REGISTRY[cls_name].load(path)


def load_fast_tokenizer():
    """Load the FAST tokenizer source from HuggingFace."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )


@register_tokenizer
class FASTTokenizerWrapper(ActionTokenizer):
    """
    Wraps the FAST+ tokenizer for use with ACT.
    """

    def __init__(self, tokenizer, max_token_len=128, action_dim=7, chunk_size=50):
        self.tokenizer = tokenizer
        self._max_token_len = max_token_len
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 30000
        self._pad_token_id = self._vocab_size

    @property
    def vocab_size(self): return self._vocab_size

    @property
    def max_token_len(self): return self._max_token_len

    @property
    def pad_token_id(self): return self._pad_token_id

    @property
    def action_dim(self): return self._action_dim

    @property
    def chunk_size(self): return self._chunk_size

    @classmethod
    def train_on_libero(cls, dataset_path, chunk_size=50, action_dim=7, max_token_len=128):
        """
        Train a custom FAST tokenizer on LIBERO action data.

        Args:
            dataset_path: path to LIBERO HDF5 directory or single file
            chunk_size: action chunk length (should match ACT's chunk_size)
            action_dim: action dimensionality (7 for LIBERO)
            max_token_len: max padded token sequence length

        Returns:
            FASTTokenizerWrapper instance with trained tokenizer
        """
        print("Loading FAST tokenizer source from HuggingFace...")
        tokenizer = load_fast_tokenizer()

        print(f"Collecting action chunks from {dataset_path}...")
        all_chunks = collect_action_chunks(dataset_path, chunk_size, action_dim)
        print(f"Collected {len(all_chunks)} action chunks of shape {all_chunks.shape}")

        # Normalize actions to [-1, 1] using quantiles (as FAST recommends)
        flat = all_chunks.reshape(-1, action_dim)
        q01 = np.quantile(flat, 0.01, axis=0)
        q99 = np.quantile(flat, 0.99, axis=0)
        scale = (q99 - q01) / 2.0
        scale = np.clip(scale, 1e-6, np.inf)
        offset = (q99 + q01) / 2.0

        normalized = (all_chunks - offset) / scale  # now roughly [-1, 1]
        normalized = np.clip(normalized, -1.0, 1.0)

        print("Training FAST tokenizer on action data...")
        tokenizer = tokenizer.fit(normalized)

        # Determine actual max token length from the data
        test_tokens = tokenizer(normalized[:100])
        actual_max = max(len(t) for t in test_tokens)
        # Use slightly larger than observed max for safety
        max_token_len = min(max_token_len, int(actual_max * 1.2) + 4)
        print(f"Max observed token length: {actual_max}, using max_token_len={max_token_len}")

        wrapper = cls(tokenizer, max_token_len=max_token_len,
                      action_dim=action_dim, chunk_size=chunk_size)
        # Store normalization params for encode/decode
        wrapper.action_offset = offset.astype(np.float32)
        wrapper.action_scale = scale.astype(np.float32)

        return wrapper

    @classmethod
    def load(cls, path):
        """Load a saved tokenizer wrapper."""
        import pickle
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        meta_path = os.path.join(path, 'wrapper_meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        wrapper = cls(tokenizer,
                      max_token_len=meta['max_token_len'],
                      action_dim=meta['action_dim'],
                      chunk_size=meta['chunk_size'])
        wrapper.action_offset = meta['action_offset']
        wrapper.action_scale = meta['action_scale']
        wrapper._vocab_size = meta['vocab_size']
        wrapper._pad_token_id = meta['pad_token_id']
        return wrapper

    def save(self, path):
        """Save the tokenizer and wrapper metadata."""
        import pickle
        os.makedirs(path, exist_ok=True)
        self._write_type_marker(path)
        self.tokenizer.save_pretrained(path)
        meta = {
            'max_token_len': self.max_token_len,
            'action_dim': self.action_dim,
            'chunk_size': self.chunk_size,
            'action_offset': self.action_offset,
            'action_scale': self.action_scale,
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
        }
        with open(os.path.join(path, 'wrapper_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        print(f"Saved FAST tokenizer wrapper to {path}")

    def encode(self, action_chunks):
        """
        Encode continuous action chunks to padded token sequences.

        Args:
            action_chunks: numpy array (B, chunk_size, action_dim) or (chunk_size, action_dim)

        Returns:
            tokens: LongTensor (B, max_token_len) — padded with pad_token_id
            token_lens: LongTensor (B,) — actual length of each token sequence
        """
        single = action_chunks.ndim == 2
        if single:
            action_chunks = action_chunks[np.newaxis]

        # Normalize to [-1, 1]
        normalized = (action_chunks - self.action_offset) / self.action_scale
        normalized = np.clip(normalized, -1.0, 1.0)

        # Tokenize (returns list of variable-length lists)
        token_lists = self.tokenizer(normalized)

        # Pad to fixed length
        batch_size = len(token_lists)
        padded = np.full((batch_size, self.max_token_len), self.pad_token_id, dtype=np.int64)
        lengths = np.zeros(batch_size, dtype=np.int64)

        for i, toks in enumerate(token_lists):
            tlen = min(len(toks), self.max_token_len)
            padded[i, :tlen] = toks[:tlen]
            lengths[i] = tlen

        tokens = torch.from_numpy(padded).long()
        token_lens = torch.from_numpy(lengths).long()

        if single:
            tokens = tokens.squeeze(0)
            token_lens = token_lens.squeeze(0)

        return tokens, token_lens

    def decode(self, tokens, token_lens=None):
        """
        Decode token sequences back to continuous action chunks.

        Args:
            tokens: LongTensor (B, max_token_len) or (max_token_len,)
            token_lens: LongTensor (B,) or scalar — actual token lengths (optional)

        Returns:
            actions: numpy array (B, chunk_size, action_dim)
        """
        single = tokens.dim() == 1
        if single:
            tokens = tokens.unsqueeze(0)
            if token_lens is not None:
                token_lens = token_lens.unsqueeze(0)

        tokens_np = tokens.cpu().numpy()
        batch_size = tokens_np.shape[0]

        # Strip padding for each sample
        token_lists = []
        for i in range(batch_size):
            if token_lens is not None:
                tlen = int(token_lens[i].item())
            else:
                # Find first pad token
                pad_mask = tokens_np[i] == self.pad_token_id
                tlen = np.argmax(pad_mask) if pad_mask.any() else len(tokens_np[i])
            token_lists.append(tokens_np[i, :tlen].tolist())

        # Decode with FAST
        decoded = self.tokenizer.decode(token_lists)  # list of numpy arrays

        # Denormalize back to original action space
        actions = []
        for arr in decoded:
            # arr shape: (chunk_size, action_dim) — but may differ slightly
            arr = np.array(arr, dtype=np.float32)
            # Pad or truncate to chunk_size
            if arr.shape[0] < self.chunk_size:
                pad = np.zeros((self.chunk_size - arr.shape[0], self.action_dim), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            elif arr.shape[0] > self.chunk_size:
                arr = arr[:self.chunk_size]
            # Denormalize
            arr = arr * self.action_scale + self.action_offset
            actions.append(arr)

        actions = np.stack(actions, axis=0)

        if single:
            actions = actions[0]

        return actions


def collect_action_chunks(dataset_path, chunk_size, action_dim):
    """
    Collect all action chunks from a LIBERO dataset for tokenizer training.
    Each demo is split into non-overlapping chunks of size chunk_size.
    """
    if os.path.isdir(dataset_path):
        hdf5_files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
    else:
        hdf5_files = [dataset_path]

    all_chunks = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            for demo_key in sorted(f['data'].keys()):
                actions = f[f'data/{demo_key}/actions'][()]  # (T, action_dim)
                # Split into chunks
                for start in range(0, len(actions), chunk_size):
                    chunk = actions[start:start + chunk_size]
                    if chunk.shape[0] == chunk_size:
                        all_chunks.append(chunk)
                    else:
                        # Pad last chunk
                        padded = np.zeros((chunk_size, action_dim), dtype=np.float32)
                        padded[:chunk.shape[0]] = chunk
                        all_chunks.append(padded)

    return np.array(all_chunks, dtype=np.float32)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train FAST tokenizer on LIBERO data')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to LIBERO HDF5 directory')
    parser.add_argument('--save_path', type=str, default='./fast_tokenizer',
                        help='Where to save the trained tokenizer')
    parser.add_argument('--chunk_size', type=int, default=50)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--max_token_len', type=int, default=128)
    args = parser.parse_args()

    wrapper = FASTTokenizerWrapper.train_on_libero(
        args.dataset_path,
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        max_token_len=args.max_token_len,
    )
    wrapper.save(args.save_path)

    # Quick sanity check
    print("\n--- Sanity check ---")
    test_actions = collect_action_chunks(args.dataset_path, args.chunk_size, args.action_dim)[:5]
    tokens, lens = wrapper.encode(test_actions)
    print(f"Encoded 5 chunks → tokens shape: {tokens.shape}, lengths: {lens.tolist()}")
    decoded = wrapper.decode(tokens, lens)
    print(f"Decoded → actions shape: {decoded.shape}")
    mse = np.mean((test_actions - decoded) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")