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

    def __init__(self, tokenizer, max_token_len=128, action_dim=7, chunk_size=50,
                 fast_scale=10, fast_min_token=0):
        self.tokenizer = tokenizer
        self._max_token_len = max_token_len
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 30000
        self._pad_token_id = self._vocab_size
        self.fast_scale = fast_scale
        self.fast_min_token = fast_min_token

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

        # Extract scale/min_token from the fitted UniversalActionProcessor,
        # then unwrap to the raw bpe_tokenizer so save/load is stable.
        fast_scale = tokenizer.scale
        fast_min_token = tokenizer.min_token
        bpe_tokenizer = tokenizer.bpe_tokenizer

        wrapper = cls(bpe_tokenizer, max_token_len=max_token_len,
                      action_dim=action_dim, chunk_size=chunk_size,
                      fast_scale=fast_scale, fast_min_token=fast_min_token)
        # Store normalization params for encode/decode
        wrapper.action_offset = offset.astype(np.float32)
        wrapper.action_scale = scale.astype(np.float32)

        return wrapper

    @classmethod
    def load(cls, path):
        """Load a saved tokenizer wrapper."""
        import pickle
        from transformers import AutoTokenizer
        # Load the raw BPE tokenizer directly — no custom code needed.
        tokenizer = AutoTokenizer.from_pretrained(path)
        meta_path = os.path.join(path, 'wrapper_meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        wrapper = cls(tokenizer,
                      max_token_len=meta['max_token_len'],
                      action_dim=meta['action_dim'],
                      chunk_size=meta['chunk_size'],
                      fast_scale=meta.get('fast_scale', 10),
                      fast_min_token=meta.get('fast_min_token', 0))
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
            'fast_scale': self.fast_scale,
            'fast_min_token': self.fast_min_token,
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

        from scipy.fft import dct as scipy_dct

        # Normalize to [-1, 1]
        normalized = (action_chunks - self.action_offset) / self.action_scale
        normalized = np.clip(normalized, -1.0, 1.0)

        # Replicate UniversalActionProcessor.__call__: DCT -> round -> chr -> BPE
        dct_coeff = scipy_dct(normalized, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.fast_scale)
        token_lists = []
        for elem in dct_coeff:
            token_str = "".join(
                map(chr, np.maximum(elem.flatten() - self.fast_min_token, 0).astype(int))
            )
            token_lists.append(self.tokenizer(token_str)["input_ids"])

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

        from scipy.fft import idct as scipy_idct

        # Replicate UniversalActionProcessor.decode: BPE decode -> ord -> IDCT -> denorm
        actions = []
        for toks in token_lists:
            try:
                text = self.tokenizer.decode(toks)
                dct_coeff = np.array(list(map(ord, text)), dtype=np.float32) + self.fast_min_token
                dct_coeff = dct_coeff.reshape(self.chunk_size, self.action_dim)
                arr = scipy_idct(dct_coeff / self.fast_scale, axis=0, norm="ortho")
                arr = arr.astype(np.float32)
            except Exception:
                arr = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            # Denormalize back to original action space
            arr = arr * self.action_scale + self.action_offset
            actions.append(arr)

        actions = np.stack(actions, axis=0)

        if single:
            actions = actions[0]

        return actions


@register_tokenizer
class QuESTTokenizerWrapper(ActionTokenizer):
    """
    Wraps QueST's SkillVAE (learned VQ/FSQ tokenizer) for use with ACT.

    Unlike FAST (DCT + BPE, variable-length, CPU-only), this uses a neural
    encoder-decoder with finite scalar quantization. Produces fixed-length
    token sequences of length skill_block_size // downsample_factor.
    """

    # Default SkillVAE hyperparameters (from QueST config/algo/quest.yaml)
    DEFAULT_VAE_KWARGS = dict(
        encoder_dim=256,
        decoder_dim=256,
        attn_pdrop=0.1,
        use_causal_encoder=True,
        use_causal_decoder=True,
        encoder_heads=4,
        encoder_layers=2,
        decoder_heads=4,
        decoder_layers=4,
        vq_type='fsq',
        fsq_level=None,
        codebook_dim=512,
        codebook_size=1024,
    )

    def __init__(self, skill_vae, action_offset, action_scale,
                 skill_block_size=32, downsample_factor=4, action_dim=7,
                 device='cuda'):
        from quest_modules import SkillVAE as _SkillVAE  # noqa: F401

        self.skill_vae = skill_vae.to(device).eval()
        self.action_offset = np.asarray(action_offset, dtype=np.float32)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.device = device

        # Compute actual vocab size from FSQ levels
        if skill_vae.vq_type == 'fsq':
            self._vocab_size = int(np.prod(skill_vae.fsq_level))
        else:
            self._vocab_size = skill_vae.vq.codebook_size

        self._max_token_len = skill_block_size // downsample_factor
        self._pad_token_id = self._vocab_size
        self._action_dim = action_dim
        self._chunk_size = skill_block_size

        # Store constructor kwargs for save/load
        self.vae_kwargs = dict(
            action_dim=action_dim,
            skill_block_size=skill_block_size,
            downsample_factor=downsample_factor,
            encoder_dim=skill_vae.encoder_dim,
            decoder_dim=skill_vae.decoder_dim,
            attn_pdrop=0.1,
            use_causal_encoder=skill_vae.use_causal_encoder,
            use_causal_decoder=skill_vae.use_causal_decoder,
            encoder_heads=skill_vae.encoder.layers[0].self_attn.num_heads,
            encoder_layers=len(skill_vae.encoder.layers),
            decoder_heads=skill_vae.decoder.layers[0].self_attn.num_heads,
            decoder_layers=len(skill_vae.decoder.layers),
            vq_type=skill_vae.vq_type,
            fsq_level=skill_vae.fsq_level,
            codebook_dim=getattr(skill_vae.vq, 'codebook_dim', 512),
            codebook_size=self._vocab_size if skill_vae.vq_type == 'fsq' else skill_vae.vq.codebook_size,
        )

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
    def train_on_libero(cls, dataset_path, chunk_size=32, action_dim=7,
                        downsample_factor=4, codebook_size=1024,
                        num_epochs=100, batch_size=256, lr=1e-4,
                        weight_decay=1e-4, device='cuda', **vae_overrides):
        """
        Train a SkillVAE tokenizer on LIBERO action data.

        Args:
            dataset_path: path to LIBERO HDF5 directory or single file
            chunk_size: action chunk length (= skill_block_size)
            action_dim: action dimensionality (7 for LIBERO)
            downsample_factor: temporal compression factor
            codebook_size: VQ/FSQ codebook size
            num_epochs: training epochs (100 recommended for LIBERO)
            batch_size: training batch size
            lr: learning rate
            weight_decay: Adam weight decay
            device: 'cuda' or 'cpu'
            **vae_overrides: override any DEFAULT_VAE_KWARGS

        Returns:
            QuESTTokenizerWrapper instance with trained SkillVAE
        """
        from quest_modules import SkillVAE
        from torch.utils.data import DataLoader, TensorDataset

        print(f"Collecting action chunks from {dataset_path}...")
        all_chunks = collect_action_chunks(dataset_path, chunk_size, action_dim)
        print(f"Collected {len(all_chunks)} action chunks of shape {all_chunks.shape}")

        # Quantile-based normalization to [-1, 1] (same as FAST)
        flat = all_chunks.reshape(-1, action_dim)
        q01 = np.quantile(flat, 0.01, axis=0)
        q99 = np.quantile(flat, 0.99, axis=0)
        scale = np.clip((q99 - q01) / 2.0, 1e-6, np.inf)
        offset = (q99 + q01) / 2.0

        normalized = np.clip((all_chunks - offset) / scale, -1.0, 1.0)

        # Build SkillVAE
        vae_kwargs = dict(cls.DEFAULT_VAE_KWARGS)
        vae_kwargs.update(
            action_dim=action_dim,
            skill_block_size=chunk_size,
            downsample_factor=downsample_factor,
            codebook_size=codebook_size,
        )
        vae_kwargs.update(vae_overrides)

        print(f"Building SkillVAE: block_size={chunk_size}, downsample={downsample_factor}, "
              f"codebook_size={codebook_size}, vq_type={vae_kwargs['vq_type']}")
        skill_vae = SkillVAE(**vae_kwargs).to(device)
        n_params = sum(p.numel() for p in skill_vae.parameters())
        print(f"SkillVAE parameters: {n_params / 1e6:.2f}M")

        # Training loop
        dataset = TensorDataset(torch.from_numpy(normalized).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(skill_vae.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        loss_fn = torch.nn.L1Loss()

        print(f"Training SkillVAE for {num_epochs} epochs...")
        skill_vae.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_pp = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(device)
                recon, pp, pp_sample, commitment_loss, _ = skill_vae(batch)
                loss = loss_fn(recon, batch) + commitment_loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_pp += pp.item()
                n_batches += 1
            avg_loss = epoch_loss / n_batches
            avg_pp = epoch_pp / n_batches
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {epoch:3d}/{num_epochs}: loss={avg_loss:.6f}, perplexity={avg_pp:.3f}")

        skill_vae.eval()
        print("SkillVAE training complete.")

        wrapper = cls(skill_vae, offset, scale,
                      skill_block_size=chunk_size,
                      downsample_factor=downsample_factor,
                      action_dim=action_dim, device=device)
        return wrapper

    @classmethod
    def from_pretrained(cls, checkpoint_path, dataset_path, chunk_size=32,
                        action_dim=7, downsample_factor=4, codebook_size=1024,
                        device='cuda', **vae_overrides):
        """
        Load a pre-trained SkillVAE from a QueST checkpoint.

        Args:
            checkpoint_path: path to QueST .pth checkpoint (e.g. multitask_model.pth)
            dataset_path: path to LIBERO HDF5 data (needed to compute normalization stats)
            chunk_size: skill_block_size used during QueST training (default 32)
            action_dim: action dimensionality (default 7)
            downsample_factor: temporal downsampling (default 4)
            codebook_size: VQ/FSQ codebook size (default 1024)
            device: 'cuda' or 'cpu'
            **vae_overrides: override any DEFAULT_VAE_KWARGS

        Returns:
            QuESTTokenizerWrapper instance with the pre-trained SkillVAE
        """
        from quest_modules import SkillVAE

        # Build SkillVAE with matching architecture
        vae_kwargs = dict(cls.DEFAULT_VAE_KWARGS)
        vae_kwargs.update(
            action_dim=action_dim,
            skill_block_size=chunk_size,
            downsample_factor=downsample_factor,
            codebook_size=codebook_size,
        )
        vae_kwargs.update(vae_overrides)
        skill_vae = SkillVAE(**vae_kwargs)

        # Load checkpoint and extract autoencoder weights
        print(f"Loading QueST checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # QueST checkpoints store model state under 'model' key
        # with autoencoder weights prefixed 'autoencoder.'
        if 'model' in ckpt:
            full_state = ckpt['model']
        else:
            full_state = ckpt

        prefix = 'autoencoder.'
        vae_state = {k[len(prefix):]: v for k, v in full_state.items()
                     if k.startswith(prefix)}

        if not vae_state:
            # Maybe it's already just the autoencoder state_dict
            vae_state = full_state

        skill_vae.load_state_dict(vae_state)
        print(f"Loaded SkillVAE weights ({len(vae_state)} parameters)")

        # Compute normalization stats from the dataset
        print(f"Computing normalization stats from {dataset_path}...")
        all_chunks = collect_action_chunks(dataset_path, chunk_size, action_dim)
        flat = all_chunks.reshape(-1, action_dim)
        q01 = np.quantile(flat, 0.01, axis=0)
        q99 = np.quantile(flat, 0.99, axis=0)
        scale = np.clip((q99 - q01) / 2.0, 1e-6, np.inf)
        offset = (q99 + q01) / 2.0

        wrapper = cls(skill_vae, offset, scale,
                      skill_block_size=chunk_size,
                      downsample_factor=downsample_factor,
                      action_dim=action_dim, device=device)
        return wrapper

    @classmethod
    def load(cls, path, device='cuda'):
        """Load a saved QuEST tokenizer wrapper."""
        import pickle
        from quest_modules import SkillVAE

        meta_path = os.path.join(path, 'wrapper_meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        skill_vae = SkillVAE(**meta['vae_kwargs'])
        state_dict = torch.load(os.path.join(path, 'skill_vae.pt'),
                                map_location=device, weights_only=True)
        skill_vae.load_state_dict(state_dict)

        wrapper = cls(skill_vae,
                      action_offset=meta['action_offset'],
                      action_scale=meta['action_scale'],
                      skill_block_size=meta['chunk_size'],
                      downsample_factor=meta['vae_kwargs']['downsample_factor'],
                      action_dim=meta['action_dim'],
                      device=device)
        return wrapper

    def save(self, path):
        """Save the SkillVAE and wrapper metadata."""
        import pickle
        os.makedirs(path, exist_ok=True)
        self._write_type_marker(path)
        torch.save(self.skill_vae.state_dict(), os.path.join(path, 'skill_vae.pt'))
        meta = {
            'vae_kwargs': self.vae_kwargs,
            'action_offset': self.action_offset,
            'action_scale': self.action_scale,
            'vocab_size': self._vocab_size,
            'pad_token_id': self._pad_token_id,
            'max_token_len': self._max_token_len,
            'action_dim': self._action_dim,
            'chunk_size': self._chunk_size,
        }
        with open(os.path.join(path, 'wrapper_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        print(f"Saved QuEST tokenizer wrapper to {path}")

    def encode(self, action_chunks):
        """
        Encode continuous action chunks to token indices.

        Args:
            action_chunks: numpy array (B, chunk_size, action_dim) or (chunk_size, action_dim)

        Returns:
            tokens:     LongTensor (B, max_token_len) — discrete token IDs
            token_lens: LongTensor (B,) — always max_token_len (fixed-length)
        """
        single = action_chunks.ndim == 2
        if single:
            action_chunks = action_chunks[np.newaxis]

        # Normalize to [-1, 1]
        normalized = (action_chunks - self.action_offset) / self.action_scale
        normalized = np.clip(normalized, -1.0, 1.0)

        tensor = torch.from_numpy(normalized).float().to(self.device)
        with torch.no_grad():
            indices = self.skill_vae.get_indices(tensor)  # (B, num_tokens)

        tokens = indices.cpu().long()
        batch_size = tokens.shape[0]
        token_lens = torch.full((batch_size,), self._max_token_len, dtype=torch.long)

        if single:
            tokens = tokens.squeeze(0)
            token_lens = token_lens.squeeze(0)

        return tokens, token_lens

    def decode(self, tokens, token_lens=None):
        """
        Decode token indices back to continuous action chunks.

        Args:
            tokens:     LongTensor (B, max_token_len) or (max_token_len,)
            token_lens: ignored (fixed-length sequences), accepted for interface compat

        Returns:
            actions: numpy array (B, chunk_size, action_dim)
        """
        single = tokens.dim() == 1
        if single:
            tokens = tokens.unsqueeze(0)

        tokens = tokens.to(self.device)
        with torch.no_grad():
            actions = self.skill_vae.decode_actions(tokens)  # (B, chunk_size, action_dim)

        actions = actions.cpu().numpy()
        # Denormalize
        actions = actions * self.action_scale + self.action_offset

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

    parser = argparse.ArgumentParser(description='Train or convert action tokenizers for ACT')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- train: train a tokenizer from scratch ---
    train_parser = subparsers.add_parser('train', help='Train a tokenizer from scratch')
    train_parser.add_argument('--tokenizer_type', type=str, default='fast',
                              choices=['fast', 'quest'],
                              help='Tokenizer type to train')
    train_parser.add_argument('--dataset_path', type=str, required=True,
                              help='Path to LIBERO HDF5 directory')
    train_parser.add_argument('--save_path', type=str, default=None,
                              help='Where to save (default: ./<type>_tokenizer)')
    train_parser.add_argument('--chunk_size', type=int, default=None,
                              help='Action chunk size (default: 50 for FAST, 32 for QueST)')
    train_parser.add_argument('--action_dim', type=int, default=7)
    # FAST-specific
    train_parser.add_argument('--max_token_len', type=int, default=128,
                              help='(FAST) Max padded token sequence length')
    # QueST-specific
    train_parser.add_argument('--downsample_factor', type=int, default=4,
                              help='(QueST) Temporal downsampling factor')
    train_parser.add_argument('--codebook_size', type=int, default=1024,
                              help='(QueST) VQ/FSQ codebook size')
    train_parser.add_argument('--quest_epochs', type=int, default=100,
                              help='(QueST) Training epochs')

    # --- convert: convert a pre-trained QueST checkpoint ---
    convert_parser = subparsers.add_parser('convert',
                                           help='Convert a pre-trained QueST checkpoint')
    convert_parser.add_argument('--checkpoint_path', type=str, required=True,
                                help='Path to QueST .pth checkpoint')
    convert_parser.add_argument('--dataset_path', type=str, required=True,
                                help='Path to LIBERO HDF5 directory (for normalization stats)')
    convert_parser.add_argument('--save_path', type=str, default='./quest_tokenizer',
                                help='Where to save converted tokenizer')
    convert_parser.add_argument('--chunk_size', type=int, default=32,
                                help='skill_block_size used during QueST training')
    convert_parser.add_argument('--action_dim', type=int, default=7)
    convert_parser.add_argument('--downsample_factor', type=int, default=4)
    convert_parser.add_argument('--codebook_size', type=int, default=1024)

    args = parser.parse_args()

    if args.command == 'train':
        if args.save_path is None:
            args.save_path = f'./{args.tokenizer_type}_tokenizer'
        if args.chunk_size is None:
            args.chunk_size = 32 if args.tokenizer_type == 'quest' else 50

        if args.tokenizer_type == 'fast':
            wrapper = FASTTokenizerWrapper.train_on_libero(
                args.dataset_path,
                chunk_size=args.chunk_size,
                action_dim=args.action_dim,
                max_token_len=args.max_token_len,
            )
        elif args.tokenizer_type == 'quest':
            wrapper = QuESTTokenizerWrapper.train_on_libero(
                args.dataset_path,
                chunk_size=args.chunk_size,
                action_dim=args.action_dim,
                downsample_factor=args.downsample_factor,
                codebook_size=args.codebook_size,
                num_epochs=args.quest_epochs,
            )

    elif args.command == 'convert':
        wrapper = QuESTTokenizerWrapper.from_pretrained(
            checkpoint_path=args.checkpoint_path,
            dataset_path=args.dataset_path,
            chunk_size=args.chunk_size,
            action_dim=args.action_dim,
            downsample_factor=args.downsample_factor,
            codebook_size=args.codebook_size,
        )

    wrapper.save(args.save_path)

    # Quick sanity check
    print("\n--- Sanity check ---")
    chunk_size = args.chunk_size if args.command == 'convert' else args.chunk_size
    test_actions = collect_action_chunks(args.dataset_path, chunk_size, args.action_dim)[:5]
    tokens, lens = wrapper.encode(test_actions)
    print(f"Encoded 5 chunks -> tokens shape: {tokens.shape}, lengths: {lens.tolist()}")
    decoded = wrapper.decode(tokens, lens)
    print(f"Decoded -> actions shape: {decoded.shape}")
    mse = np.mean((test_actions - decoded) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")