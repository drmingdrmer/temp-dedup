#!/usr/bin/env python3
"""
File deduplication tool using hash-based indexing and MinHash for similarity detection.
"""

import os
import hashlib
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set
from minhash import MinHashSignature, create_signature
import k3fmt
import k3color


class UStr(object):
    def __init__(self, s):
        self.s = s

    def __len__(self):
        # calcuate the width, unicode counts twice as an ascii.
        width = 0 
        for k in self.s:
            if ord(k) < 128:
                width += 1
            else:
                width += 2
        return width

    def __str__(self):
        return self.s

    def trim(self, max: int) -> 'UStr':
        if self.__len__() <= max:
            return self

        width = 0
        res = ""
        for x in self.s:
            if ord(x) < 128:
                width += 1
            else:
                width += 2
            if width > max - 3:
                return UStr(res + "..")
            res += x

        return UStr(res)



class FileIndex:
    """File and directory indexing system for deduplication."""

    def __init__(self, args):
        self.args = args
        self.base_path = os.path.abspath(args.path)
        self.dedup_dir = os.path.join(self.base_path, ".dedup")
        self.index_path = os.path.join(self.dedup_dir, "file_index.yaml")
        self.minhash_path = os.path.join(self.dedup_dir, "minhash_index.yaml")

        os.makedirs(self.dedup_dir, exist_ok=True)

        self.hash_to_paths: Dict[str, List[str]] = {}
        self.path_to_minhash: Dict[str, List[int]] = {}

        if args.load_existing:
            self._load_indexes()

    def _load_indexes(self) -> None:
        """Load existing indexes from disk."""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                self.hash_to_paths = data.get('hash_to_paths', {})

        if os.path.exists(self.minhash_path):
            with open(self.minhash_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                self.path_to_minhash = data.get('path_to_minhash', {})

    def _save_indexes(self) -> None:
        """Save indexes to disk."""
        # Save hash index with metadata
        hash_data = {
            '_meta': {
                'total_hashes': len(self.hash_to_paths),
                'total_paths': sum(len(paths) for paths in self.hash_to_paths.values()),
                'duplicates': len([h for h, paths in self.hash_to_paths.items() if len(paths) > 1])
            },
            'hash_to_paths': self.hash_to_paths
        }

        with open(self.index_path, 'w', encoding='utf-8') as f:
            yaml.dump(hash_data, f, default_flow_style=False, allow_unicode=True,
                     sort_keys=True, indent=2)

        # Save minhash index with metadata
        minhash_data = {
            '_meta': {
                'total_directories': len(self.path_to_minhash),
                'bucket_size': 128
            },
            'path_to_minhash': self.path_to_minhash
        }

        with open(self.minhash_path, 'w', encoding='utf-8') as f:
            yaml.dump(minhash_data, f, default_flow_style=False, allow_unicode=True,
                     sort_keys=True, indent=2)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash for a file using filename and size."""
        stat = os.stat(file_path)
        filename = os.path.basename(file_path)
        data = f"{filename}:{stat.st_size}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _compute_dir_hash(self, dir_path: str) -> str:
        """Compute hash for a directory recursively."""
        hash_parts = []

        for item in sorted(os.listdir(dir_path)):
            if item == ".dedup":
                continue

            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                file_hash = self._compute_file_hash(item_path)
                hash_parts.append(file_hash)
            elif os.path.isdir(item_path):
                dir_hash = self._compute_dir_hash(item_path)
                hash_parts.append(dir_hash)


        combined = ":".join(hash_parts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _compute_dir_minhash(self, dir_path: str) -> MinHashSignature:
        """Compute MinHash signature for a directory based on direct children."""
        elements = []

        for item in sorted(os.listdir(dir_path)):
            if item == ".dedup":
                continue

            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                stat = os.stat(item_path)
                elements.append(f"{item}:{stat.st_size}")
            elif os.path.isdir(item_path):
                elements.append(f"{item}:dir")

        return create_signature(elements, buckets=128)

    def build_index(self) -> None:
        """Index files and directories in the base path."""
        if os.path.isfile(self.base_path):
            self._index_file(self.base_path)
        elif os.path.isdir(self.base_path):
            self._index_directory(self.base_path)

        self._save_indexes()

    def _index_file(self, file_path: str) -> None:
        """Index a single file."""
        file_hash = self._compute_file_hash(file_path)
        if file_hash not in self.hash_to_paths:
            self.hash_to_paths[file_hash] = []

        if file_path not in self.hash_to_paths[file_hash]:
            self.hash_to_paths[file_hash].append(file_path)

    def _index_directory(self, dir_path: str) -> None:
        """Index a directory recursively."""
        for root, dirs, files in os.walk(dir_path):
            # Skip .dedup directory
            if ".dedup" in dirs:
                dirs.remove(".dedup")

            # Index all files
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self._index_file(file_path)
                except (OSError, IOError) as e:
                    print(f"Warning: Failed to index file {file_path}: {e}")

            # Index the current directory
            try:
                dir_hash = self._compute_dir_hash(root)
                if dir_hash not in self.hash_to_paths:
                    self.hash_to_paths[dir_hash] = []

                if root not in self.hash_to_paths[dir_hash]:
                    self.hash_to_paths[dir_hash].append(root)
            except (OSError, IOError, ValueError) as e:
                print(f"Warning: Failed to hash directory {root}: {e}")

            # Compute MinHash for the directory
            try:
                minhash_sig = self._compute_dir_minhash(root)
                self.path_to_minhash[root] = [h for h in minhash_sig.min_hashes if h is not None]
            except (OSError, IOError, ValueError) as e:
                print(f"Warning: Failed to compute MinHash for {root}: {e}")

    def find_duplicates(self) -> Dict[str, List[str]]:
        """Find duplicate files and directories based on hash values."""
        duplicates = {}

        for file_hash, paths in self.hash_to_paths.items():
            if len(paths) > 1:
                duplicates[file_hash] = paths

        return duplicates

    def find_similar_directories(self) -> List[tuple]:
        """Find similar directories using MinHash with given threshold."""
        threshold = self.args.threshold
        similar_pairs = []
        dir_paths = list(self.path_to_minhash.keys())

        for i in range(len(dir_paths)):
            for j in range(i + 1, len(dir_paths)):
                path_a, path_b = dir_paths[i], dir_paths[j]

                # Reconstruct MinHash signatures
                sig_a = MinHashSignature(128)
                sig_a.min_hashes = [h if h != 0 else None for h in self.path_to_minhash[path_a]]
                sig_a.min_hashes += [None] * (128 - len(sig_a.min_hashes))

                sig_b = MinHashSignature(128)
                sig_b.min_hashes = [h if h != 0 else None for h in self.path_to_minhash[path_b]]
                sig_b.min_hashes += [None] * (128 - len(sig_b.min_hashes))

                similarity = sig_a.compute_similarity(sig_b)

                if similarity >= threshold:
                    similar_pairs.append((path_a, path_b, similarity))

        return similar_pairs

    def print_duplicates(self) -> None:
        """Print found duplicates."""
        duplicates = self.find_duplicates()

        if not duplicates:
            print("No duplicates found.")
            return

        print(f"Found {len(duplicates)} groups of duplicates:")
        for file_hash, paths in duplicates.items():
            print(f"\nHash: {file_hash[:16]}...")
            for path in paths:
                print(f"  {path}")

    def print_similar_directories(self) -> None:
        """Print similar directories."""
        similar = self.find_similar_directories()
        threshold = self.args.threshold

        if not similar:
            print(f"No similar directories found (threshold: {threshold:.2%}).")
            return

        print(f"Found {len(similar)} pairs of similar directories (threshold: {threshold:.2%}):")
        for path_a, path_b, similarity in similar:
            print(f"\nSimilarity: {similarity:.2%}")
            print(f"  {path_a}")
            print(f"  {path_b}")

    def remove_directory(self, dir_path: str) -> None:
        """Remove a directory and update indexes."""
        # Remove from hash_to_paths
        try:
            dir_hash = self._compute_dir_hash(dir_path)
            if dir_hash in self.hash_to_paths:
                if dir_path in self.hash_to_paths[dir_hash]:
                    self.hash_to_paths[dir_hash].remove(dir_path)
                    if not self.hash_to_paths[dir_hash]:
                        del self.hash_to_paths[dir_hash]
        except (OSError, IOError, ValueError):
            pass  # Directory may already be removed or inaccessible

        # Remove from path_to_minhash
        if dir_path in self.path_to_minhash:
            del self.path_to_minhash[dir_path]

        # Actually delete the directory
        shutil.rmtree(dir_path)

        # Save updated indexes
        self._save_indexes()

    def find_directory_duplicates(self, current_dir: str) -> List[str]:
        """Find duplicates of a specific directory."""
        try:
            dir_hash = self._compute_dir_hash(current_dir)
            if dir_hash not in self.hash_to_paths:
                return []

            duplicates = [p for p in self.hash_to_paths[dir_hash] if p != current_dir]
            return duplicates
        except (OSError, IOError, ValueError):
            return []

    def interactive_cleanup(self) -> None:
        """Interactive cleanup process starting from top-level directories."""
        print(f"Starting interactive cleanup of: {self.base_path}")
        print("Processing directories from top-level first...\n")

        # Get all directories sorted by depth (top-level first)
        all_dirs = []
        for root, dirs, files in os.walk(self.base_path):
            # Skip .dedup directory
            if ".dedup" in dirs:
                dirs.remove(".dedup")

            if root != self.base_path:  # Skip the base path itself
                depth = root.replace(self.base_path, "").count(os.sep)
                all_dirs.append((depth, root))

        # Sort by depth (ascending - top level first)
        all_dirs.sort(key=lambda x: x[0])

        processed = set()

        for depth, current_dir in all_dirs:
            if current_dir in processed or not os.path.exists(current_dir):
                continue

            duplicates = self.find_directory_duplicates(current_dir)
            if not duplicates:
                continue

            valid_duplicates = []
            for dup_path in duplicates:
                if os.path.exists(dup_path) and dup_path not in processed:
                    valid_duplicates.append(dup_path)

            if not valid_duplicates:
                continue

            # Include current_dir in the list for unified presentation
            all_duplicates = [current_dir] + valid_duplicates

            print(f"\n=== Found duplicate directories ===")
            for i, path in enumerate(all_duplicates):
                self._print_directory_context(i, path)

            print(f"  [s] Skip this group")

            print(f"\nWhich directory do you want to KEEP? (Enter number 1-{len(all_duplicates)} or 's' to skip)")

            while True:
                try:
                    choice = input("Choice: ").strip()

                    if choice == 's':
                        break
                    else:
                        idx = int(choice) - 1
                        if 0 <= idx < len(all_duplicates):
                            # Keep the selected directory, remove all others
                            keep_path = all_duplicates[idx]
                            to_remove = [p for p in all_duplicates if p != keep_path]

                            print(f"\nKeeping: {keep_path}")
                            print("Removing:")
                            for rem_path in to_remove:
                                print(f"  {rem_path}")
                                try:
                                    self.remove_directory(rem_path)
                                    processed.add(rem_path)
                                    print(f"  ✓ Successfully removed: {rem_path}")
                                except (OSError, IOError) as e:
                                    print(f"  ✗ Failed to remove {rem_path}: {e}")
                            processed.add(keep_path)
                            break
                        else:
                            print(f"Invalid choice. Please enter 1-{len(all_duplicates)} or 's'")
                except ValueError:
                    print(f"Invalid input. Please enter a number (1-{len(all_duplicates)}) or 's'")
                except KeyboardInterrupt:
                    print("\nCleanup interrupted by user.")
                    return

        print("\nInteractive cleanup completed.")

    def _get_path_segments(self, dir_path: str) -> List[str]:
        """Get path components relative to base_path."""
        if not dir_path.startswith(self.base_path):
            return []

        rel_path = os.path.relpath(dir_path, self.base_path)
        if rel_path == '.':
            return []

        return rel_path.split(os.sep)

    def _get_sibling_context(self, parent_path: str, part: str) -> List[UStr]:
        """Get 7-line context for a path component (3 before + current + 3 after)."""

        part_width = UStr(part).__len__()

        siblings = sorted([item for item in os.listdir(parent_path)
                         if item != '.dedup' and os.path.exists(os.path.join(parent_path, item))])

        idx = siblings.index(part)

        # Get before items (3 before current)
        before_items = siblings[max(0, idx - 3):idx]
        before_lines = []
        if max(0, idx - 3) > 0:
            before_lines.append("   ...")
        for item in before_items:
            item_path = os.path.join(parent_path, item)
            before_lines.append(f"   {item}")

        # Pad before_lines to exactly 3 items
        while len(before_lines) < 4:
            before_lines.insert(0, "")

        # Current item (highlight with brackets)
        current_line = f" / {part}"

        # Get after items (3 after current)
        after_items = siblings[idx + 1:min(len(siblings), idx + 4)]
        after_lines = []
        for item in after_items:
            item_path = os.path.join(parent_path, item)
            after_lines.append(f"   {item}")

        if idx + 4 < len(siblings):
            after_lines.append("   ...")

        # Pad after_lines to exactly 3 items
        while len(after_lines) < 4:
            after_lines.append("")

        res = before_lines + [current_line] + after_lines
        # "/" is 1 width
        return [UStr(x).trim(part_width+3) for x in res]


    def _format_and_print_segments(self, i: int, segments: List[List[UStr]], dir_path: str) -> None:
        """Format and print segments using k3fmt."""
        if not segments or not any(segments):
            return

        print()  # Empty line before context
        result = k3fmt.format_line(segments, sep='', aligns='llllllllllllllllll')
        result = result.split('\n')
        for idx, line in enumerate(result):
            if idx == 4:
                print(k3color.white(f"  [{i+1}] {line}"))
            else:
                print(k3color.cyan("      " + line))
        print()

    def _print_directory_context(self, i: int, dir_path: str) -> None:
        """Print directory context showing surrounding files/dirs at each level."""
        path_parts = self._get_path_segments(dir_path)
        if not path_parts:
            return

        # Collect context for each level
        segments = [['', '', '', '', self.base_path, '', '', '', '']]
        parent_path = self.base_path

        for part in path_parts:

            seg = self._get_sibling_context(parent_path, part)
            segments.append(seg)

            parent_path = os.path.join(parent_path, part)

        # Print context using k3fmt
        self._format_and_print_segments(i, segments, dir_path)



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive file deduplication tool")
    parser.add_argument("path", help="Directory path to scan and deduplicate")
    parser.add_argument("--load-existing", action="store_true",
                       help="Load existing index instead of rebuilding")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Similarity threshold for directory comparison (default: 0.8)")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        return

    if not os.path.isdir(args.path):
        print(f"Error: Path is not a directory: {args.path}")
        return

    index = FileIndex(args)

    if args.load_existing:
        print(f"Loading existing index for: {args.path}")
        if not os.path.exists(index.index_path):
            print("No existing index found. Building new index...")
            index.build_index()
        else:
            print("Index loaded successfully.")
    else:
        print(f"Building index for: {args.path}")
        index.build_index()
        print("Index completed.")

    # Start interactive cleanup
    index.interactive_cleanup()


if __name__ == "__main__":
    main()
