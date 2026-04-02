import os
import sys
import re
import random
import argparse
import pandas as pd
from typing import List, Dict, Set

class CorpusAdapter:
    """
    Corpus Template Adapter for ASR Testing.
    Supports both complex nested slot templates and standard text corpus.
    """
    def __init__(self, excel_path: str, target_count: int = 1000):
        self.excel_path = os.path.abspath(excel_path)
        self.target_count = target_count
        self.slot_dict: Dict[str, List[str]] = {}
        self.templates: List[str] = []
        self.sent_list: List[str] = []
        self.is_standard_corpus = False

    def parse_excel(self):
        try:
            xl = pd.ExcelFile(self.excel_path)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel file {self.excel_path}: {e}", file=sys.stderr)
            sys.exit(1)

        has_special_sheet = False
        
        for sheet in xl.sheet_names:
            sheet_lower = sheet.lower()
            
            # 1. Parse static sentences
            if 'sent' in sheet_lower:
                has_special_sheet = True
                raw_sents = pd.read_excel(xl, sheet_name=sheet, header=None).values.flatten()
                self.sent_list.extend([str(x).strip() for x in raw_sents if pd.notna(x) and str(x).strip()])
                
            # 2. Parse templates
            elif 'shuofa' in sheet_lower:
                has_special_sheet = True
                raw_tpls = pd.read_excel(xl, sheet_name=sheet, header=None).values.flatten()
                self.templates.extend([str(x).strip() for x in raw_tpls if pd.notna(x) and str(x).strip()])
                
            # 3. Parse slot dictionaries
            elif '<>' in sheet:
                has_special_sheet = True
                df_slot = pd.read_excel(xl, sheet_name=sheet)
                for col in df_slot.columns:
                    col_name = str(col).strip()
                    if col_name.startswith('<') and col_name.endswith('>'):
                        self.slot_dict[col_name] = df_slot[col].dropna().astype(str).str.strip().tolist()

        # [Fallback] If no specific sheets found, treat as standard corpus
        if not has_special_sheet:
            self.is_standard_corpus = True
            df = pd.read_excel(xl, sheet_name=0)
            if 'text' in df.columns:
                raw_sents = df['text'].dropna().astype(str).str.strip().tolist()
            else:
                raw_sents = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            self.sent_list.extend(raw_sents)

    def _expand_template(self, template: str) -> str:
        """Recursively expand slots in a template."""
        result = template
        max_depth = 20
        depth = 0
        
        while re.search(r'<[^>]+>', result) and depth < max_depth:
            match = re.search(r'<[^>]+>', result)
            slot_name = match.group(0)
            
            if slot_name in self.slot_dict and self.slot_dict[slot_name]:
                val = random.choice(self.slot_dict[slot_name])
                result = result[:match.start()] + str(val) + result[match.end():]
            else:
                # Break if slot is not defined in dictionary to avoid infinite loop
                break
            depth += 1
            
        return result

    def generate_testset(self) -> List[str]:
        """Generate text set based on parsed rules or fallback strategy."""
        final_set: Set[str] = set()
        
        # Fallback pipeline for standard corpus
        if self.is_standard_corpus or (not self.templates and self.sent_list):
            final_list = list(set(self.sent_list))
            if len(final_list) > self.target_count:
                return random.sample(final_list, self.target_count)
            return final_list
            
        # Priority sampling from 'sent'
        if len(self.sent_list) >= self.target_count:
            return random.sample(self.sent_list, self.target_count)
            
        final_set.update(self.sent_list)
        needed = self.target_count - len(final_set)
        
        # Slot expansion
        attempts = 0
        max_attempts = needed * 15
        
        while len(final_set) < self.target_count and attempts < max_attempts and self.templates:
            generated = self._expand_template(random.choice(self.templates))
            # Only add successfully expanded sentences
            if '<' not in generated:
                final_set.add(generated)
            attempts += 1
            
        final_list = list(final_set)
        if len(final_list) > self.target_count:
            final_list = random.sample(final_list, self.target_count)
            
        random.shuffle(final_list)
        return final_list

def main():
    parser = argparse.ArgumentParser(description="Corpus Template Adapter for ASR")
    parser.add_argument("-i", "--input", required=True, help="Input Excel corpus file path")
    parser.add_argument("-o", "--output", required=True, help="Output flat txt file path")
    parser.add_argument("-n", "--num", type=int, default=1000, help="Target number of sentences to generate")
    args = parser.parse_args()

    adapter = CorpusAdapter(args.input, target_count=args.num)
    adapter.parse_excel()
    generated_sents = adapter.generate_testset()

    if not generated_sents:
        print(f"[WARNING] No text could be generated from {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for sent in generated_sents:
            f.write(f"{sent}\n")

    print(f"[INFO] Successfully generated {len(generated_sents)} sentences to {args.output}")

if __name__ == "__main__":
    main()