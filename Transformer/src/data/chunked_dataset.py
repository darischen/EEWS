"""
Chunked Dataset for streaming large sequences from disk
Inspired by Mini-Stockfish's approach to handling 316M lines
"""

import numpy as np
import os
import logging
import json
from torch.utils.data import Dataset
import gc

logger = logging.getLogger(__name__)


class ChunkedSequenceDataset(Dataset):
    """
    Streams sequences from disk in chunks instead of loading all into RAM.
    Saves memory by only keeping current batch in memory.
    Implements sequential chunk caching for optimal disk I/O.
    """

    def __init__(self, X, y_1d, y_5d, y_20d, y_vol, chunk_dir='data/processed', chunk_size=500000):
        """
        Args:
            X, y_*: Full sequence arrays (will be saved to disk in chunks)
            chunk_dir: Directory to save chunks
            chunk_size: Sequences per chunk file
        """
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        os.makedirs(chunk_dir, exist_ok=True)

        self.total_samples = len(X)
        self.chunk_indices = []
        self.chunk_files = []

        # Store shape metadata
        self.X_shape = X.shape
        self.metadata_file = os.path.join(chunk_dir, 'metadata.json')

        # Multi-chunk caching: keep recent chunks fully loaded in RAM (not lazy memmaps)
        self._chunk_cache = {}  # Dict of {chunk_id: {arrays_dict}} - fully in RAM
        self._cache_order = []  # Track LRU order
        self._max_cached_chunks = 5  # Keep up to 5 chunks in RAM (for shuffle + prefetch)

        # Check if chunks already exist with matching metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            if (metadata.get('X_shape') == list(X.shape) and
                metadata.get('total_samples') == self.total_samples and
                metadata.get('chunk_size') == chunk_size):
                logger.info(f"Loading existing {self.total_samples:,} sequences from chunks...")
                self._load_existing_chunks()
                return

        logger.info(f"Creating {self.total_samples:,} sequences in chunks of {chunk_size:,}...")

        # Save sequences in chunks
        num_chunks = (self.total_samples + chunk_size - 1) // chunk_size

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, self.total_samples)

            # Save each chunk
            chunk_prefix = os.path.join(chunk_dir, f'chunk_{chunk_id:05d}')

            # Save as numpy memmap files for efficient access
            X_chunk = np.memmap(chunk_prefix + '_X.npy', dtype=np.float32, mode='w+',
                               shape=(end_idx - start_idx, X.shape[1], X.shape[2]))
            X_chunk[:] = X[start_idx:end_idx]
            X_chunk.flush()

            y_1d_chunk = np.memmap(chunk_prefix + '_y1d.npy', dtype=np.float32, mode='w+',
                                   shape=(end_idx - start_idx,))
            y_1d_chunk[:] = y_1d[start_idx:end_idx]
            y_1d_chunk.flush()

            y_5d_chunk = np.memmap(chunk_prefix + '_y5d.npy', dtype=np.float32, mode='w+',
                                   shape=(end_idx - start_idx,))
            y_5d_chunk[:] = y_5d[start_idx:end_idx]
            y_5d_chunk.flush()

            y_20d_chunk = np.memmap(chunk_prefix + '_y20d.npy', dtype=np.float32, mode='w+',
                                    shape=(end_idx - start_idx,))
            y_20d_chunk[:] = y_20d[start_idx:end_idx]
            y_20d_chunk.flush()

            y_vol_chunk = np.memmap(chunk_prefix + '_vol.npy', dtype=np.float32, mode='w+',
                                    shape=(end_idx - start_idx,))
            y_vol_chunk[:] = y_vol[start_idx:end_idx]
            y_vol_chunk.flush()

            self.chunk_indices.append((chunk_id, start_idx, end_idx))
            self.chunk_files.append((chunk_prefix + '_X.npy',
                                    chunk_prefix + '_y1d.npy',
                                    chunk_prefix + '_y5d.npy',
                                    chunk_prefix + '_y20d.npy',
                                    chunk_prefix + '_vol.npy'))

            del X_chunk, y_1d_chunk, y_5d_chunk, y_20d_chunk, y_vol_chunk
            gc.collect()

            logger.info(f"Saved chunk {chunk_id + 1}/{num_chunks} ({end_idx - start_idx:,} samples)")

        # Save metadata with shapes
        metadata = {
            'X_shape': list(X.shape),
            'total_samples': self.total_samples,
            'chunk_size': chunk_size
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)

    def _load_existing_chunks(self):
        """Load chunk metadata from existing files"""
        chunk_id = 0
        while True:
            chunk_prefix = os.path.join(self.chunk_dir, f'chunk_{chunk_id:05d}')
            X_file = chunk_prefix + '_X.npy'
            if not os.path.exists(X_file):
                break

            start_idx = chunk_id * self.chunk_size
            end_idx = min((chunk_id + 1) * self.chunk_size, self.total_samples)

            self.chunk_indices.append((chunk_id, start_idx, end_idx))
            self.chunk_files.append((X_file,
                                    chunk_prefix + '_y1d.npy',
                                    chunk_prefix + '_y5d.npy',
                                    chunk_prefix + '_y20d.npy',
                                    chunk_prefix + '_vol.npy'))
            chunk_id += 1

        logger.info(f"Loaded {len(self.chunk_indices)} existing chunks")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """Get single sample by index with multi-chunk prefetching cache"""
        # Find which chunk contains this index
        chunk_id, start_idx, end_idx = None, None, None
        for cid, sid, eid in self.chunk_indices:
            if sid <= idx < eid:
                chunk_id, start_idx, end_idx = cid, sid, eid
                break

        if chunk_id is None:
            raise IndexError(f"Index {idx} out of range")

        local_idx = idx - start_idx
        chunk_size = end_idx - start_idx

        # Load chunk fully into RAM if not in cache
        if chunk_id not in self._chunk_cache:
            files = self.chunk_files[chunk_id]

            # Load memmaps and convert to actual arrays in RAM (not lazy)
            arrays = {
                'X': np.asarray(np.memmap(files[0], dtype=np.float32, mode='r',
                              shape=(chunk_size, self.X_shape[1], self.X_shape[2]))),
                'y_1d': np.asarray(np.memmap(files[1], dtype=np.float32, mode='r',
                                 shape=(chunk_size,))),
                'y_5d': np.asarray(np.memmap(files[2], dtype=np.float32, mode='r',
                                 shape=(chunk_size,))),
                'y_20d': np.asarray(np.memmap(files[3], dtype=np.float32, mode='r',
                                  shape=(chunk_size,))),
                'y_vol': np.asarray(np.memmap(files[4], dtype=np.float32, mode='r',
                                  shape=(chunk_size,)))
            }
            self._chunk_cache[chunk_id] = arrays
            self._cache_order.append(chunk_id)

            # Evict oldest chunk if cache is full
            if len(self._chunk_cache) > self._max_cached_chunks:
                oldest_id = self._cache_order.pop(0)
                del self._chunk_cache[oldest_id]
        else:
            # Move accessed chunk to end of LRU order
            if chunk_id in self._cache_order:
                self._cache_order.remove(chunk_id)
                self._cache_order.append(chunk_id)

        # Extract sample from cached chunk (now in RAM, no disk access)
        arrays = self._chunk_cache[chunk_id]
        X = arrays['X'][local_idx].copy()
        y_1d = arrays['y_1d'][local_idx].copy()
        y_5d = arrays['y_5d'][local_idx].copy()
        y_20d = arrays['y_20d'][local_idx].copy()
        y_vol = arrays['y_vol'][local_idx].copy()

        return X, y_1d, y_5d, y_20d, y_vol
