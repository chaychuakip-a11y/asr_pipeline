import os
import json
import pandas as pd
import hashlib
import argparse
from datetime import datetime
from typing import Dict

def get_semantic_hash(file_path: str) -> str:
    """
    Calculate Semantic Hash.
    Priority 1: 'sent', 'shuofa', '<>' sheets (Template format).
    Priority 2: 'text' column or first column in Sheet 0 (Standard format).
    """
    try:
        import pandas as pd
        import hashlib
        import os
        
        xl = pd.ExcelFile(file_path)
        content_list = []
        has_special_sheet = False
        
        for sheet in xl.sheet_names:
            sheet_lower = sheet.lower()
            if 'sent' in sheet_lower or 'shuofa' in sheet_lower:
                has_special_sheet = True
                df = pd.read_excel(xl, sheet_name=sheet, header=None)
                vals = df.values.flatten()
                content_list.extend([str(x).strip() for x in vals if pd.notna(x) and str(x).strip()])
                
            elif '<>' in sheet:
                has_special_sheet = True
                df = pd.read_excel(xl, sheet_name=sheet)
                for col in df.columns:
                    col_name = str(col).strip()
                    if col_name.startswith('<') and col_name.endswith('>'):
                        vals = df[col].dropna().astype(str).str.strip().tolist()
                        content_list.extend([col_name] + vals)
                        
        # [Fallback Logic] Standard Corpus Compatibility
        if not has_special_sheet:
            df = pd.read_excel(xl, sheet_name=0)
            if 'text' in df.columns:
                vals = df['text'].dropna().astype(str).str.strip().tolist()
                content_list.extend(vals)
            else:
                vals = df.values.flatten()
                content_list.extend([str(x).strip() for x in vals if pd.notna(x) and str(x).strip()])
                
        if not content_list:
            raise ValueError("No valid text content found in any sheet.")
            
        content_str = "|".join(content_list)
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
        
    except Exception as e:
        import hashlib
        import os
        print(f"[Warning] Semantic hash failed for {os.path.basename(file_path)}: {str(e)}. Fallback to File MD5.")
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

def warmup_manifest(corpus_dir: str, manifest_path: str, msg: str):
    """
    Scan corpus directory and populate the manifest with semantic hashes.
    """
    processed_state: Dict[str, dict] = {}
    
    # Load existing manifest if it exists
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                processed_state = json.load(f)
        except json.JSONDecodeError:
            print(f"[Warning] Failed to decode existing manifest at {manifest_path}. Creating a new one.")

    print(f"Scanning historical corpus for project [{msg}] at: {corpus_dir}")
    count = 0
    skip_count = 0
    
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                file_path = os.path.join(root, file)
                content_hash = get_semantic_hash(file_path)
                file_key = file # 使用文件名作为追踪主键
                
                # 检查文件名是否已记录，且 hash 未发生变化
                if file_key not in processed_state or processed_state[file_key].get("hash") != content_hash:
                    processed_state[file_key] = {
                        "hash": content_hash,
                        "file_path": os.path.abspath(file_path),
                        "task_msg": msg,
                        "processed_time": "WARMUP_INITIALIZED_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    count += 1
                    print(f"  -> Indexed: {file} | Hash: {content_hash[:8]}...")
                else:
                    skip_count += 1

    # Persist the state back to JSON
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(processed_state, f, indent=4, ensure_ascii=False)
    
    print("\n=== Warm-up Complete ===")
    print(f"Project: {msg}")
    print(f"New Files Indexed: {count}")
    print(f"Files Skipped (Already Tracked): {skip_count}")
    print(f"Manifest Path: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Initialize tracking manifest for historical ASR corpus.")
    parser.add_argument("-c", "--corpus_dir", required=True, help="Absolute path to the project corpus directory")
    parser.add_argument("-m", "--manifest_path", required=True, help="Path to save the output manifest.json")
    parser.add_argument("--msg", required=True, help="Task identifier (e.g., aweita)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.corpus_dir):
        print(f"Error: Corpus directory not found at {args.corpus_dir}")
        sys.exit(1)
        
    warmup_manifest(args.corpus_dir, args.manifest_path, args.msg)

if __name__ == "__main__":
    main()