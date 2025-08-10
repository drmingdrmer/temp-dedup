#!/usr/bin/env python3
"""
MinHash implementation for computing set similarity using constant space.
"""

import hashlib
from typing import List, Set, Union


class MinHashSignature:
    """MinHash signature for a set of elements."""
    
    def __init__(self, buckets: int = 128):
        if buckets <= 0:
            raise ValueError("buckets must be positive")
        self.buckets = buckets
        self.min_hashes = [None] * buckets
    
    def add_element(self, element: str) -> None:
        """Add an element to the signature."""
        hash_val = int(hashlib.sha1(element.encode('utf-8')).hexdigest(), 16)
        bucket_id = hash_val % self.buckets
        
        if self.min_hashes[bucket_id] is None or hash_val < self.min_hashes[bucket_id]:
            self.min_hashes[bucket_id] = hash_val
    
    def compute_similarity(self, other: 'MinHashSignature') -> float:
        """Compute Jaccard similarity with another signature."""
        if self.buckets != other.buckets:
            raise ValueError("signatures must have same number of buckets")
        
        matches = 0
        valid_buckets = 0
        
        for i in range(self.buckets):
            # Only count buckets where at least one signature has a value
            if self.min_hashes[i] is not None or other.min_hashes[i] is not None:
                valid_buckets += 1
                if self.min_hashes[i] == other.min_hashes[i]:
                    matches += 1
        
        if valid_buckets == 0:
            return 0.0
        
        return matches / valid_buckets
    
    def __str__(self) -> str:
        """String representation of the signature for debugging."""
        non_empty = [(i, h) for i, h in enumerate(self.min_hashes) if h is not None]
        return f"MinHashSignature(buckets={self.buckets}, non_empty={len(non_empty)}, hashes={non_empty[:10]}{'...' if len(non_empty) > 10 else ''})"


def create_signature(elements: Union[List[str], Set[str]], buckets: int = 128) -> MinHashSignature:
    """Create MinHash signature for a collection of elements."""
    signature = MinHashSignature(buckets)
    for element in elements:
        signature.add_element(element)
    return signature


def compute_similarity(set_a: Union[List[str], Set[str]], 
                      set_b: Union[List[str], Set[str]], 
                      buckets: int = 128) -> float:
    """Compute Jaccard similarity between two sets using MinHash."""
    sig_a = create_signature(set_a, buckets)
    sig_b = create_signature(set_b, buckets)
    return sig_a.compute_similarity(sig_b)


if __name__ == "__main__":
    # Test 1: Small dataset (original test case)
    print("=== Small Dataset Test ===")
    set_a = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]
    set_b = ["file2.txt", "file3.txt", "file5.txt", "file6.txt"]
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Intersection: {len(set(set_a) & set(set_b))} files")
    print(f"Union: {len(set(set_a) | set(set_b))} files")
    
    sig_a = create_signature(set_a, buckets=128)
    sig_b = create_signature(set_b, buckets=128)
    print("sig_a:", sig_a)
    print("sig_b:", sig_b)
    
    similarity = sig_a.compute_similarity(sig_b)
    actual_similarity = len(set(set_a) & set(set_b)) / len(set(set_a) | set(set_b))
    print(f"MinHash similarity: {similarity:.2%}")
    print(f"Actual similarity: {actual_similarity:.2%}")
    print()
    
    # Test 2: Larger dataset
    print("=== Large Dataset Test ===")
    import random
    
    # Generate larger test sets
    base_set = [f"file{i:06d}.txt" for i in range(1000)]
    random.shuffle(base_set)
    
    set_a = base_set[:400]
    set_b = base_set[200:600]  # 200 overlap, 200 unique each
    
    print(f"Set A: {len(set_a)} files")
    print(f"Set B: {len(set_b)} files") 
    print(f"Intersection: {len(set(set_a) & set(set_b))} files")
    print(f"Union: {len(set(set_a) | set(set_b))} files")
    
    sig_a = create_signature(set_a, buckets=64)
    sig_b = create_signature(set_b, buckets=64)
    
    similarity = sig_a.compute_similarity(sig_b)
    actual_similarity = len(set(set_a) & set(set_b)) / len(set(set_a) | set(set_b))
    print(f"MinHash similarity: {similarity:.2%}")
    print(f"Actual similarity: {actual_similarity:.2%}")
