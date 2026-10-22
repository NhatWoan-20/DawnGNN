"""
Dataset class for CNN+BiGRU malware detection.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
import numpy as np


class APISequenceDataset(Dataset):
    """Dataset for API sequences."""
    
    def __init__(
        self,
        sequences_path: Path,
        labels_path: Path,
        word_to_idx_path: Path,
        split_path: Path = None
    ):
        """
        Initialize dataset.
        
        Args:
            sequences_path: Path to api_sequences.json
            labels_path: Path to labels.json
            word_to_idx_path: Path to word_to_idx.json
            split_path: Optional path to split JSON (list of SHA256)
        """
        # Load sequences
        with open(sequences_path, 'r', encoding='utf-8') as f:
            self.sequences = json.load(f)
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        # Load word to index mapping
        with open(word_to_idx_path, 'r', encoding='utf-8') as f:
            self.word_to_idx = json.load(f)
        
        # Filter by split if provided
        if split_path:
            with open(split_path, 'r') as f:
                split_shas = set(json.load(f))
            self.shas = [sha for sha in self.sequences.keys() if sha in split_shas]
        else:
            self.shas = list(self.sequences.keys())
        
        # Unknown token index (for APIs not in vocabulary)
        # Padding is at index 0, so unk_idx should be max_idx + 1
        max_idx = max(self.word_to_idx.values()) if self.word_to_idx else 0
        self.unk_idx = max_idx + 1
    
    def __len__(self):
        return len(self.shas)
    
    def __getitem__(self, idx):
        sha = self.shas[idx]
        sequence = self.sequences[sha]
        label = self.labels.get(sha, 0)
        
        # Convert sequence to indices
        indices = []
        for api in sequence:
            if api == "<PAD>":
                idx_val = 0  # Padding token
            else:
                idx_val = self.word_to_idx.get(api, self.unk_idx)
            indices.append(idx_val)
        
        return {
            'sequence': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'sha': sha
        }

