import os
import sys
import shutil
import hashlib
import argparse
import glob
from datetime import datetime
from typing import Set

class LexiconVCS:
    def __init__(self, dict_path: str, max_versions: int = 10):
        self.dict_path = os.path.abspath(dict_path)
        self.work_dir = os.path.dirname(self.dict_path)
        self.dict_name = os.path.basename(self.dict_path)
        self.history_dir = os.path.join(self.work_dir, ".history")
        self.log_file = os.path.join(self.history_dir, "history.log")
        self.max_versions = max_versions
        
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

    def _get_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of the given file."""
        if not os.path.exists(file_path):
            return "0000000"
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:7]

    def _get_latest_backup(self) -> str:
        """Find the most recent backup file in the history directory."""
        pattern = os.path.join(self.history_dir, f"{self.dict_name}.v*.bak")
        backups = sorted(glob.glob(pattern))
        return backups[-1] if backups else ""

    def _load_vocab(self, file_path: str) -> Set[str]:
        """Load dictionary words into a set for fast diffing."""
        vocab = set()
        if not os.path.exists(file_path):
            return vocab
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    vocab.add(parts[0])
        return vocab

    def pre_merge(self) -> bool:
        """Create a snapshot before any modifications."""
        # Ensure target file exists to prevent read errors later
        if not os.path.exists(self.dict_path):
            open(self.dict_path, 'w', encoding='utf-8').close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = self._get_md5(self.dict_path)
        backup_name = f"{self.dict_name}.v{timestamp}.{file_hash}.bak"
        backup_path = os.path.join(self.history_dir, backup_name)

        shutil.copy2(self.dict_path, backup_path)
        print(f"Pre-merge snapshot created: {backup_path}")
        return True

    def post_merge(self, task_msg: str, lang_id: str) -> bool:
        """Calculate diff, write detailed log, and prune old versions."""
        latest_bak = self._get_latest_backup()
        if not latest_bak:
            print("Error: No pre-merge backup found to compare against.", file=sys.stderr)
            return False

        # Calculate exact differences
        old_vocab = self._load_vocab(latest_bak)
        new_vocab = self._load_vocab(self.dict_path)
        
        added_words = new_vocab - old_vocab
        added_count = len(added_words)
        total_count = len(new_vocab)
        
        new_hash = self._get_md5(self.dict_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Detailed log entry
        log_entry = (
            f"[{timestamp}] TASK: {task_msg} | LANG: {lang_id} | "
            f"HASH: {new_hash} | TOTAL_WORDS: {total_count} | "
            f"ADDED_WORDS: {added_count}\n"
        )
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
            
        print(f"Post-merge log updated. Added {added_count} words.")
        
        # Trigger pruning to maintain size limits
        self._prune()
        return True

    def _prune(self):
        """Keep only the latest N backups and truncate log file accordingly."""
        pattern = os.path.join(self.history_dir, f"{self.dict_name}.v*.bak")
        backups = sorted(glob.glob(pattern))
        
        if len(backups) <= self.max_versions:
            return
            
        # Remove oldest backups
        backups_to_delete = backups[:-self.max_versions]
        for bak in backups_to_delete:
            os.remove(bak)
            
        # Truncate history.log to keep only recent entries (proportional to max_versions)
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            lines_to_keep = self.max_versions * 2 
            if len(lines) > lines_to_keep:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-lines_to_keep:])
                    
        print(f"Pruned {len(backups_to_delete)} old snapshots. Kept latest {self.max_versions}.")

    def log(self):
        """Display version history."""
        if not os.path.exists(self.log_file):
            print("No version history found.")
            return

        print(f"\n=== Version History for {self.dict_name} ===")
        with open(self.log_file, "r", encoding="utf-8") as f:
            print(f.read().strip())
        print("=========================================\n")

    def rollback(self, target_hash: str) -> bool:
        """Restore dictionary to a specific historical hash version."""
        pattern = os.path.join(self.history_dir, f"{self.dict_name}.v*.{target_hash}.bak")
        matches = glob.glob(pattern)

        if not matches:
            print(f"Error: No snapshot found matching hash '{target_hash}'", file=sys.stderr)
            return False

        if len(matches) > 1:
            print(f"Error: Multiple snapshots found for hash '{target_hash}'. Collision detected.", file=sys.stderr)
            return False

        backup_path = matches[0]
        
        # Auto-commit current state before overriding
        self.pre_merge()

        shutil.copy2(backup_path, self.dict_path)
        print(f"VCS Rollback successful. Restored to hash: {target_hash}")
        
        # Record rollback action in log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] ROLLBACK | Restored to Hash: {target_hash}\n")
            
        return True

def main():
    parser = argparse.ArgumentParser(description="Lexicon Version Control System CLI")
    parser.add_argument("-i", "--input", type=str, required=True, help="Absolute path to the dictionary file (e.g., res/arabic_res/ubctc_duan/new_dict)")
    parser.add_argument("action", choices=["pre_merge", "post_merge", "rollback", "log"], help="Lifecycle or intervention action")
    parser.add_argument("-m", "--msg", type=str, default="UnknownTask", help="Task message identifier (for post_merge)")
    parser.add_argument("-l", "--lang", type=str, default="0", help="Language ID (for post_merge)")
    parser.add_argument("-t", "--target_hash", type=str, help="Target MD5 hash to restore (for rollback action)")
    parser.add_argument("--max_versions", type=int, default=10, help="Max history versions to retain")

    args = parser.parse_args()
    vcs = LexiconVCS(args.input, args.max_versions)

    if args.action == "pre_merge":
        vcs.pre_merge()
    elif args.action == "post_merge":
        vcs.post_merge(args.msg, args.lang)
    elif args.action == "log":
        vcs.log()
    elif args.action == "rollback":
        if not args.target_hash:
            print("Error: --target_hash (-t) is required for rollback action.", file=sys.stderr)
            sys.exit(1)
        vcs.rollback(args.target_hash)

if __name__ == "__main__":
    main()