import argparse
import sys
import os

def load_valid_phones(syms_path: str) -> set:
    """Load valid phonemes from phones.syms for validation."""
    valid_phones = set()
    if not os.path.exists(syms_path):
        print(f"Warning: phones.syms not found at {syms_path}. Phone validation skipped.", file=sys.stderr)
        return valid_phones
    
    with open(syms_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                valid_phones.add(parts[0])
    return valid_phones

def merge_dictionaries(base_dict_path: str, new_dict_path: str, syms_path: str):
    valid_phones = load_valid_phones(syms_path)
    existing_words = set()
    base_lines = []
    
    # 1. Load base dictionary to memory, preserving order
    if os.path.exists(base_dict_path):
        with open(base_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\r\n')
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 1:
                    existing_words.add(parts[0])
                base_lines.append(line)
                
    # 2. Parse and validate new dictionary
    added_count = 0
    skip_duplicate = 0
    skip_format = 0
    skip_abnormal_phone = 0
    
    with open(new_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
                
            # Check format: Strictly word\tphone1 phone2
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Format Error (Skipped): '{line}'", file=sys.stderr)
                skip_format += 1
                continue
                
            word, phones_str = parts[0], parts[1].strip()
            
            # Deduplication
            if word in existing_words:
                skip_duplicate += 1
                continue
                
            # Phoneme Validation
            phones = phones_str.split(' ')
            if valid_phones:
                invalid_phones = [p for p in phones if p not in valid_phones]
                if invalid_phones:
                    print(f"Abnormal Phone {invalid_phones} in word '{word}'. Skipped.", file=sys.stderr)
                    skip_abnormal_phone += 1
                    continue
            
            # Append valid entry
            base_lines.append(f"{word}\t{' '.join(phones)}")
            existing_words.add(word)
            added_count += 1
            
    # 3. Write back with strict Linux newlines (\n)
    with open(base_dict_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(base_lines) + '\n')
        
    print(f"Merge complete. Added: {added_count} | Duplicates: {skip_duplicate} | "
          f"Format Errors: {skip_format} | Abnormal Phones: {skip_abnormal_phone}")

def main():
    parser = argparse.ArgumentParser(description="Strict Lexicon Merge Tool")
    parser.add_argument("-i", "--input_new", required=True, help="Path to new dictionary (G2P output)")
    parser.add_argument("-o", "--output_base", required=True, help="Path to base dictionary (res/.../new_dict)")
    parser.add_argument("-p", "--phone_syms", required=True, help="Path to phones.syms for validation")
    args = parser.parse_args()
    
    merge_dictionaries(args.output_base, args.input_new, args.phone_syms)

if __name__ == "__main__":
    main()