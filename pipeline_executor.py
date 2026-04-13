"""
asr pipeline executor

asr automated pipeline executor, supporting:
- phase 1: resource build & g2p
- phase 2: incremental testset generation
- phase 3: baseline evaluation
"""

import argparse
import configparser
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import pandas as pd
import yaml

# =============================================================================
# utility functions
# =============================================================================

def get_file_md5_suffix(file_path: str, suffix_len: int = 4) -> str:
    """calculate md5 of a file and return the last n characters."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[-suffix_len:]


def load_language_map(file_path: str) -> dict:
    """load language mapping configuration."""
    if not os.path.exists(file_path):
        print(f"[error] language map file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    lang_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            lang_map[k.strip()] = v.strip().lower()
    return lang_map


def run_subprocess(cmd: List[str], cwd: str, log_file: str, env: Optional[Dict[str, str]] = None) -> bool:
    """execute subprocess, capture logs, and optionally inject environment variables."""
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] CMD: {' '.join(cmd)}\n")
            f.flush()
            process = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
            process.wait()
            return process.returncode == 0
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\n[fatal error] {str(e)}\n")
        return False


def resolve_and_bind_paths(global_cfg: dict, base_path: str) -> dict:
    """resolve relative paths in configuration to absolute paths."""
    py_exec = global_cfg.get('python_exec')
    if not py_exec:
        global_cfg['python_exec'] = sys.executable
    elif not os.path.isabs(py_exec):
        global_cfg['python_exec'] = os.path.abspath(os.path.join(base_path, py_exec))

    local_keys = [
        'g2p_root_dir', 'tools_dir', 'merge_dict_script',
        'adapter_script', 'test_script'
    ]
    for key in local_keys:
        val = global_cfg.get(key)
        if val and not os.path.isabs(val):
            global_cfg[key] = os.path.abspath(os.path.join(base_path, val))

    return global_cfg


# =============================================================================
# delta tracker logic
# =============================================================================

class DeltaTracker:
    """semantic hash-based delta tracker for incremental processing."""

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.history = self._load_manifest()

    def _load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def get_semantic_hash(file_path: str) -> str:
        """calculate semantic hash of the excel corpus."""
        try:
            import pandas as pd
            import hashlib
            
            xl = pd.ExcelFile(file_path)
            content_list = []
            has_special_sheet = False

            for sheet in xl.sheet_names:
                sheet_lower = sheet.lower()
                
                if 'sent' in sheet_lower or 'shuofa' in sheet_lower:
                    has_special_sheet = True
                    df = pd.read_excel(xl, sheet_name=sheet, header=None)
                    vals = df.values.flatten()
                    content_list.extend([
                        str(x).strip() for x in vals
                        if pd.notna(x) and str(x).strip()
                    ])
                    
                elif '<>' in sheet:
                    has_special_sheet = True
                    df = pd.read_excel(xl, sheet_name=sheet)
                    for col in df.columns:
                        col_name = str(col).strip()
                        if col_name.startswith('<') and col_name.endswith('>'):
                            vals = df[col].dropna().astype(str).str.strip().tolist()
                            content_list.extend([col_name] + vals)

            if not has_special_sheet:
                df = pd.read_excel(xl, sheet_name=0)
                if 'text' in df.columns:
                    vals = df['text'].dropna().astype(str).str.strip().tolist()
                    content_list.extend(vals)
                else:
                    vals = df.values.flatten()
                    content_list.extend([str(x).strip() for x in vals if pd.notna(x) and str(x).strip()])

            if not content_list:
                raise ValueError("no valid text content found.")

            content_str = "|".join(content_list)
            return hashlib.md5(content_str.encode('utf-8')).hexdigest()

        except Exception:
            import hashlib
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

    def save(self):
        """persist state to json manifest."""
        Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
            
    def update_history(self, file_path: str, new_hash: str):
        """update hash record for a specific file."""
        file_key = os.path.basename(file_path)
        self.history[file_key] = {
            "hash": new_hash,
            "processed_time": datetime.now().strftime("%Y%m%d_%H%M%S")
        }


# =============================================================================
# phase 1: resource build & g2p
# =============================================================================
def generate_context_for_hebrew_oov(oov_file_path: str, corpus_dir: str, g2p_input_txt: str, target_encoding: str, target_newline: str):
    """Generate context sentences for Hebrew OOV words from Excel corpus."""
    with open(oov_file_path, 'r', encoding='utf-8') as f:
        oov_list = [line.strip() for line in f if line.strip()]

    if not oov_list:
        return

    sent_list = []
    shuofa_list = []
    slot_dict = {}

    # Parse Excel Corpus
    if os.path.exists(corpus_dir):
        for file in os.listdir(corpus_dir):
            if not file.endswith('.xlsx') or file.startswith('~'):
                continue
            try:
                xl = pd.ExcelFile(os.path.join(corpus_dir, file))
                has_special = False
                for sheet in xl.sheet_names:
                    sheet_lower = sheet.lower()
                    if 'sent' in sheet_lower:
                        has_special = True
                        df = pd.read_excel(xl, sheet_name=sheet, header=None)
                        sent_list.extend([str(x).strip() for x in df.values.flatten() if pd.notna(x)])
                    elif 'shuofa' in sheet_lower:
                        has_special = True
                        df = pd.read_excel(xl, sheet_name=sheet, header=None)
                        shuofa_list.extend([str(x).strip() for x in df.values.flatten() if pd.notna(x)])
                    elif '<>' in sheet:
                        has_special = True
                        df = pd.read_excel(xl, sheet_name=sheet)
                        for col in df.columns:
                            col_name = str(col).strip()
                            if col_name.startswith('<') and col_name.endswith('>'):
                                if col_name not in slot_dict:
                                    slot_dict[col_name] = []
                                slot_dict[col_name].extend(df[col].dropna().astype(str).str.strip().tolist())
                if not has_special:
                    df = pd.read_excel(xl, sheet_name=0)
                    if 'text' in df.columns:
                        sent_list.extend(df['text'].dropna().astype(str).str.strip().tolist())
                    else:
                        sent_list.extend(df.iloc[:, 0].dropna().astype(str).str.strip().tolist())
            except Exception:
                continue

    output_lines = []
    for oov in oov_list:
        matched = False
        
        # Priority 1: Search in standard sentences (sent)
        for sent in sent_list:
            if oov in sent:
                output_lines.append(sent)
                matched = True
                break
        if matched: continue

        # Priority 2: Search in slots and map to shuofa
        for slot_name, slot_values in slot_dict.items():
            if oov in slot_values:
                valid_shuofas = [s for s in shuofa_list if slot_name in s]
                if valid_shuofas:
                    chosen_shuofa = random.choice(valid_shuofas)
                    context_sent = chosen_shuofa.replace(slot_name, oov)
                    # Fill other slots with random valid values
                    for other_slot, other_values in slot_dict.items():
                        if other_slot != slot_name and other_slot in context_sent and other_values:
                            context_sent = context_sent.replace(other_slot, random.choice(other_values))
                    # Clean up unreplaced slots
                    context_sent = re.sub(r'<[^>]+>', '', context_sent)
                    output_lines.append(context_sent)
                    matched = True
                    break
        
        # Fallback: Write OOV directly if no context found
        if not matched:
            output_lines.append(oov)

    with open(g2p_input_txt, 'w', encoding=target_encoding, newline=target_newline) as f_out:
        for line in output_lines:
            f_out.write(f"{line}{target_newline}")



def build_base_command(task: dict, python_exec: str, train_script: str, asrmlg_exp_dir: str) -> List[str]:
    """build base command with dynamic parameter pass-through."""
    base_cmd = [
        python_exec if train_script.endswith('.py') else "bash",
        train_script
    ]

    single_dash_whitelist = {'l', 'G', 'cp', 'np'}
    internal_keys = {
        'enable_g2p', 'enable_merge_dict', 'enable_testset', 'enable_eval',
        'input_wav', 'output', 'only_corpus_process', 'enable_whisper_package', 'whisper_config'
    }

    path_keys = {
        'excel_corpus_path', 'norm_excel_corpus_path',
        'norm_train_data_slot', 'norm_train_data_shuofa', 'np',
        'train_data_slot', 'train_data_shuofa', 'cp',
        'word_syms', 'phone_syms', 'triphone_syms', 'dict',
        'hmm_list', 'hmm_list_blank', 'mapping'
    }

    for key, val in task.items():
        if key in internal_keys or val is None:
            continue
            
        if key in single_dash_whitelist:
            prefix = f"-{key}"
        else:
            prefix = f"--{key}" if len(key) > 1 else f"-{key}"

        if isinstance(val, bool):
            if val:
                base_cmd.append(prefix)
            continue

        if key in path_keys and isinstance(val, str) and not os.path.isabs(val):
            val = os.path.join(asrmlg_exp_dir, val)

        base_cmd.extend([prefix, str(val)])

    return base_cmd


def step1_extract_oov(base_cmd: List[str], task_out_path: str, msg: str,
                      asrmlg_exp_dir: str, log_file: str) -> bool:
    """extract oov words."""
    task_out_path_temp = task_out_path + "_temp"
    cmd = base_cmd + ["--only_corpus_process", "--output", task_out_path_temp]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: phase1_step1 (extract oov) for {msg}")
    return run_subprocess(cmd, asrmlg_exp_dir, log_file)


def step2_g2p_predict(task: dict, global_cfg: dict, msg: str, task_out_path: str, log_file: str) -> bool:
    """execute g2p prediction with exclusive process lock."""
    lang_abbr_map = global_cfg.get('lang_abbr_map', {})
    task_out_path_temp = task_out_path + "_temp"
    oov_file_path = Path(task_out_path_temp) / "custom_corpus_process" / "dict_dir" / "aaa_oov_base_dict"

    if not oov_file_path.exists() or oov_file_path.stat().st_size == 0:
        return True

    lang_id = str(task.get('l', ''))
    lang_abbr = lang_abbr_map.get(lang_id) or str(task.get('language', msg))

    g2p_lang_dir = Path(global_cfg.get('g2p_root_dir', '')) / str(lang_abbr) / "g2p_models"

    if not g2p_lang_dir.is_dir():
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[warning] g2p directory not found: {g2p_lang_dir}. skipping.\n")
        return True

    g2p_input_txt = g2p_lang_dir / "input.txt"
    g2p_output_dict_shared = g2p_lang_dir / "output.dict"
    private_output_dict = Path(task_out_path_temp) / f"g2p_output_{msg}.dict"

    target_encoding = 'utf-8'
    target_newline = '\n'
    if g2p_input_txt.exists():
        with open(g2p_input_txt, 'rb') as f_probe:
            raw_bytes = f_probe.read(2)
            if raw_bytes in (b'\xff\xfe', b'\xfe\xff'):
                target_encoding = 'utf-16'
                target_newline = '\r\n'

    print(f"[{datetime.now().strftime('%H:%M:%S')}] WAITING LOCK: phase1_step2 for {msg}")
    lock_file_path = g2p_lang_dir / ".g2p_engine_exec.lock"
    
    try:
        with open(lock_file_path, 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ACQUIRED LOCK: executing g2p for {msg}")
                
                # Apply external replacement list if provided
                replacement_path = global_cfg.get('g2p_replacement_list')
                replacements = {}
                if replacement_path and os.path.exists(replacement_path):
                    with open(replacement_path, 'r', encoding='utf-8') as f_rep:
                        for line in f_rep:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                replacements[parts[0]] = " ".join(parts[1:])

                # Check for Hebrew and apply context generation
                is_hebrew = lang_abbr and lang_abbr.lower() in ['he', 'heb', 'hebrew']
                if is_hebrew:
                    corpus_dir = os.path.join(global_cfg.get('asrmlg_exp_dir', ''), task.get('excel_corpus_path', ''))
                    generate_context_for_hebrew_oov(str(oov_file_path), corpus_dir, str(g2p_input_txt), target_encoding, target_newline)
                else:
                    with open(oov_file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        lines = f_in.readlines()
                    with open(g2p_input_txt, 'w', encoding=target_encoding, newline=target_newline) as f_out:
                        for line in lines:
                            word = line.strip()
                            if word in replacements:
                                word = replacements[word]
                            f_out.write(f"{word}{target_newline}")
                
                # Separate handling for Cloud vs Local G2P
                cloud_langs = global_cfg.get('cloud_g2p_langs', [])
                g2p_script = "run_cloud.sh" if lang_abbr in cloud_langs or lang_id in cloud_langs else "run.sh"
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] RUNNING G2P: {g2p_script} for {msg}")
                success = run_subprocess(["bash", g2p_script], str(g2p_lang_dir), log_file)
                
                if success and g2p_output_dict_shared.exists():
                    shutil.copy2(g2p_output_dict_shared, private_output_dict)
                else:
                    return False
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] RELEASED LOCK: g2p finished for {msg}")
                
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f_log:
            f_log.write(f"[error] locked g2p execution failed: {str(e)}\n")
        return False

    return True


def step3_merge_dict(task: dict, global_cfg: dict, msg: str, log_file: str) -> bool:
    """merge predicted phonemes into primary lexicon."""
    res_dir_map = global_cfg.get('res_dir_map', {})
    lang_abbr_map = global_cfg.get('lang_abbr_map', {})
    merge_script = global_cfg.get('merge_dict_script')
    res_base_dir = os.path.join(global_cfg.get('asrmlg_exp_dir'), global_cfg.get('res_dir_name', 'res'))

    if not (merge_script and os.path.exists(merge_script) and res_base_dir):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[warning] merge dependencies missing. skipping merge.\n")
        return True

    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: phase1_step3 (merge oov dict) for {msg}")

    python_exec = global_cfg.get('python_exec', 'python')
    vcs_script_path = os.path.join(global_cfg.get('tools_dir', './'), 'lexicon_vcs.py')

    lang_id = str(task.get('l', 0))
    lang_abbr = lang_abbr_map.get(lang_id)
    is_yun_val = str(task.get('is_yun', '0'))
    res_dir_name = res_dir_map.get(is_yun_val, "unknown")
    lang_map = global_cfg.get('parsed_language_map', {})
    lang_name = lang_map.get(lang_id, "")

    target_res_dir = os.path.join(res_base_dir, f"{lang_name}_res", res_dir_name)
    target_dict_path = os.path.join(target_res_dir, "new_dict")
    # [Modify] Fallback logic for phone validation files
    phone_syms_path = os.path.join(target_res_dir, "phones.syms")
    if not os.path.exists(phone_syms_path):
        fallback_path = os.path.join(target_res_dir, "phones.list.noblank")
        if os.path.exists(fallback_path):
            phone_syms_path = fallback_path
        else:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[WARNING] Neither phones.syms nor phones.list.noblank found in {target_res_dir}. Phoneme validation will be skipped.\n")

    g2p_output_dict = os.path.join(global_cfg.get('g2p_root_dir', ''), lang_abbr, "g2p_models", "output.dict")
    if not os.path.exists(g2p_output_dict):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[error] g2p_output_dict missing at {g2p_output_dict}. aborting merge.\n")
        return False

    if os.path.exists(vcs_script_path):
        run_subprocess([python_exec, vcs_script_path, "-i", target_dict_path, "pre_merge"],
                       global_cfg.get('asrmlg_exp_dir'), log_file)

    merge_cmd = [
        python_exec, merge_script,
        "-i", g2p_output_dict,
        "-o", target_dict_path,
        "-p", phone_syms_path
    ]
    if task.get('predict_phone_for_new'):
        merge_cmd.append("--predict_phone_for_new")

    merge_success = run_subprocess(merge_cmd, global_cfg.get('asrmlg_exp_dir'), log_file)

    if merge_success and os.path.exists(vcs_script_path):
        run_subprocess([
            python_exec, vcs_script_path,
            "-i", target_dict_path,
            "post_merge",
            "-m", task.get('msg', ''),
            "-l", lang_id,
            "--max_versions", str(global_cfg.get('max_versions', 10))
        ], global_cfg.get('asrmlg_exp_dir'), log_file)

    # 针对 Hebrew 的特殊处理：合并完毕后将更新的 Lexicon 拷贝回 G2P 模型目录
    is_hebrew = lang_abbr and lang_abbr.lower() in ['he', 'heb', 'hebrew']
    if is_hebrew and merge_success:
        import shutil
        g2p_lang_dir = os.path.join(global_cfg.get('g2p_root_dir', ''), lang_abbr, "g2p_models")
        if os.path.exists(g2p_lang_dir):
            shutil.copy2(target_dict_path, g2p_lang_dir)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[INFO] Copied merged dictionary back to Hebrew G2P directory: {g2p_lang_dir}\n")

    return True

def step4_full_build(base_cmd: List[str], task_out_path: str, msg: str,
                     patch_type: str, asrmlg_exp_dir: str, log_file: str) -> bool:
    """compile wfst and package standard asr model."""
    task_out_path_temp = task_out_path + "_temp"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CLEANING: temp dir")
    try:
        shutil.rmtree(task_out_path_temp, ignore_errors=True)
    except OSError as e:
        pass

    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: phase1_step4 (full resource build) for {msg}")

    base_cmd = base_cmd + ["--output", task_out_path]
    try:
        idx = base_cmd.index("--msg")
        base_cmd[idx + 1] = f"{msg}_{patch_type}"
    except (ValueError, IndexError):
        base_cmd.extend(["--msg", f"{msg}_{patch_type}"])

    return run_subprocess(base_cmd, asrmlg_exp_dir, log_file)


def check_whisper_dependencies(source_dir: str) -> bool:
    """validate if wfst artifacts are ready for whisper serialization."""
    required_artifacts = [
        os.path.join(source_dir, "custom_G_pak", "GeneratedG.DONE"),
        os.path.join(source_dir, "custom_G_pak", "G"),
        os.path.join(source_dir, "custom_corpus_process", "dict_dir", "aaa_dict_for_use")
    ]
    return all(os.path.exists(f) for f in required_artifacts)


def generate_custom_cfg(template_path: str, output_cfg_path: str, work_dir: str, bin_output_name: str, patch_scale: str):
    """inject absolute paths and exact output names into wfst_serialize config."""
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str
    config.read(template_path, encoding='utf-8')

    if 'common' in config:
        config['common']['lm_factor'] = str(patch_scale)

    abs_work_dir = os.path.abspath(work_dir)
    if 'input' in config:
        config['input']['wfst_net_txt'] = f"{abs_work_dir}/output.wfst.mvrd.txt"
        config['input']['edDcitSymsFile'] = f"{abs_work_dir}/edDictPhones.syms"
        config['input']['phoneSymsFile'] = f"{abs_work_dir}/edDictPhones.syms"
        config['input']['wordsSymsFile'] = f"{abs_work_dir}/words.syms"
        config['input']['word2PhoneFile'] = f"{abs_work_dir}/aaa_dict_for_use.remake"

    if 'output' in config:
        config['output']['OutWfst.bin'] = f"./output/{bin_output_name}"

    with open(output_cfg_path, 'w', encoding='utf-8') as f:
        config.write(f)

    return output_cfg_path


def step5_whisper_package(task: dict, global_cfg: dict, hybridcnn_gpatch: str, log_file: str) -> bool:
    """execute zero-copy concurrency pipeline for whisper wfst serialization."""
    if str(task.get('is_yun', '')) != '3':
        return True

    whisper_cfg = task.get('whisper_config', {})
    msg = str(task.get('msg', 'music'))
    lang_id = str(task.get('l', ''))
    lang_map = global_cfg.get('parsed_language_map', {})
    lang_name = lang_map.get(lang_id, f"lang_{lang_id}")
    timestamp = datetime.now().strftime('%Y%m%d')
    current_user = os.environ.get('USER', 'default_user')

    work_dir_name = whisper_cfg.get('work_dir', f"{lang_name}_{msg}_patch_{timestamp}")
    patch_name = whisper_cfg.get('name', f"{current_user}_{msg}_{timestamp}")
    patch_type = whisper_cfg.get('patch_type', msg)
    patch_scale = str(whisper_cfg.get('patch_scale', '1.0'))
    
    train_dict = whisper_cfg.get('train_dict')
    phoneset_path = whisper_cfg.get('phoneset')
    package_ed_target = whisper_cfg.get('package_ed_target')

    if not all([train_dict, phoneset_path, package_ed_target]):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n[error] missing required whisper config\n")
        return False

    if not os.path.isabs(phoneset_path):
        base_pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        phoneset_path = os.path.join(base_pipeline_dir, phoneset_path)

    whisper_tools_dir = global_cfg.get('whisper_tools_dir', global_cfg.get('asrmlg_exp_dir'))
    work_dir = os.path.join(whisper_tools_dir, work_dir_name)
    python_exec = global_cfg.get('python_exec', 'python')

    base_out_dir = global_cfg.get('output_dir')
    final_whisper_out = os.path.join(base_out_dir, lang_name, msg, f"whisper_bin_{timestamp}")
    os.makedirs(final_whisper_out, exist_ok=True)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    files_to_copy = [
        (os.path.join(hybridcnn_gpatch, "custom_G_pak", "G"), os.path.join(work_dir, "G")),
        (os.path.join(hybridcnn_gpatch, "custom_G_pak", "GeneratedG.DONE"), os.path.join(work_dir, "GeneratedG.DONE")),
        (os.path.join(hybridcnn_gpatch, "custom_corpus_process", "dict_dir", "aaa_dict_for_use"), os.path.join(work_dir, "aaa_dict_for_use"))
    ]

    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[error] missing required file: {src}\n")
            return False

    replace_dict_cmd = ["bash", "run_replace_dict.sh", train_dict, work_dir, lang_id]
    if not run_subprocess(replace_dict_cmd, whisper_tools_dir, log_file): return False

    dict_remake_path = os.path.join(work_dir, "aaa_dict_for_use.remake")
    package_ed_cmd = ["./package_ed", dict_remake_path, phoneset_path, package_ed_target, work_dir]
    if not run_subprocess(package_ed_cmd, whisper_tools_dir, log_file): return False

    wearlized_dir = os.path.join(whisper_tools_dir, "wearlized")
    wearlized_output_dir = os.path.join(wearlized_dir, "output")
    os.makedirs(wearlized_output_dir, exist_ok=True)

    task_uuid = uuid.uuid4().hex[:8]
    template_cfg = os.path.join(wearlized_dir, "wfst_serialize_large.241227_patch.cfg")
    custom_cfg_name = f"task_{task_uuid}.cfg"
    custom_cfg_path = os.path.join(wearlized_dir, custom_cfg_name)
    
    exact_bin_name = f"whisper_{patch_type}_{patch_scale}_{patch_name}_{task_uuid}.bin"
    generate_custom_cfg(template_cfg, custom_cfg_path, work_dir, exact_bin_name, patch_scale)

    try:
        run_env = os.environ.copy()
        run_env['LD_LIBRARY_PATH'] = f"./:{run_env.get('LD_LIBRARY_PATH', '')}"
        
        serialize_cmd = ["./wfst_serialize", custom_cfg_name]
        if not run_subprocess(serialize_cmd, wearlized_dir, log_file, env=run_env): 
            return False

        target_bin_file = os.path.join(wearlized_output_dir, exact_bin_name)
        
        if not os.path.exists(target_bin_file):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[error] serialized bin file not found at expected path: {target_bin_file}\n")
            return False

        try:
            md5_tail = get_file_md5_suffix(target_bin_file, 4)
            base_name, ext = os.path.splitext(exact_bin_name)
            final_bin_name = f"{base_name}_{md5_tail}{ext}"
            final_dest_path = os.path.join(final_whisper_out, final_bin_name)
            
            shutil.copy2(target_bin_file, final_dest_path)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[info] natively copied bin to {final_dest_path} (md5 tail: {md5_tail})\n")
                
        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[error] failed during native md5 calculation or copying: {str(e)}\n")
            return False

    finally:
        if os.path.exists(custom_cfg_path):
            os.remove(custom_cfg_path)
        if 'target_bin_file' in locals() and os.path.exists(target_bin_file):
            os.remove(target_bin_file)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: whisper artifacts exported to {final_whisper_out}")
    return True


def run_phase1_pipeline(task: dict, global_cfg: dict, asrmlg_exp_dir: str,
                        python_exec: str, train_script: str, log_file: str) -> bool:
    """phase 1 pipeline state machine with bypass routing."""
    msg = str(task.get('msg'))
    lang_id = str(task.get('l', ''))
    base_out_dir = global_cfg.get('output_dir')
    resolved_msg = str(task.get('msg', '')).strip()
    scheme_map = global_cfg.get('scheme_map', {})

    lang_map = global_cfg.get('parsed_language_map', {})
    lang_name = lang_map.get(lang_id, f"lang_{lang_id}")
    model_type = scheme_map.get(str(task.get('is_yun', 0)), 'unknown')
    
    whisper_cfg = task.get('whisper_config', {})
    explicit_source_dir = whisper_cfg.get('source_patch_dir')

    if explicit_source_dir:
        if check_whisper_dependencies(explicit_source_dir):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] USING EXPLICIT SOURCE: {explicit_source_dir}. skipping to whisper package.")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[info] using explicit source_patch_dir: {explicit_source_dir}. bypassing build steps.\n")
                
            if task.get('enable_whisper_package'):
                return step5_whisper_package(task, global_cfg, explicit_source_dir, log_file)
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] FATAL: explicit source_patch_dir missing required artifacts.")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[error] explicit source_patch_dir {explicit_source_dir} is invalid or incomplete.\n")
            return False

    # default target directory for new build
    target_dir = os.path.join(
        base_out_dir, lang_name, resolved_msg,
        f"{model_type}_{datetime.now().strftime('%Y%m%d')}"
    )

    base_cmd = build_base_command(task, python_exec, train_script, asrmlg_exp_dir)

    if not step1_extract_oov(base_cmd, target_dir, msg, asrmlg_exp_dir, log_file):
        return False

    if task.get('enable_g2p'):
        if not step2_g2p_predict(task, global_cfg, msg, target_dir, log_file):
            return False

    if task.get('enable_merge_dict'):
        step3_merge_dict(task, global_cfg, msg, log_file)

    is_yun_val = str(task.get('is_yun', '0'))
    patch_type = scheme_map.get(is_yun_val, "unknown")
    if not step4_full_build(base_cmd, target_dir, msg, patch_type, asrmlg_exp_dir, log_file):
        return False

    if task.get('enable_whisper_package'):
        return step5_whisper_package(task, global_cfg, target_dir, log_file)

    return True


def execute_phase1(tasks: List[dict], global_cfg: dict):
    """dispatch phase 1 tasks to process pool."""
    print("\n=== pipeline phase 1: resource build & g2p (parallel mode) ===")

    python_exec = global_cfg.get('python_exec')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir')
    train_script = os.path.join(
        asrmlg_exp_dir, global_cfg.get('train_script', 'corpus_process_package.py')
    )

    futures = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for task in tasks:
            if not any([
                task.get('enable_g2p'), 
                task.get('enable_merge_dict'), 
                task.get('enable_whisper_package')
            ]):
                continue

            lang_id = str(task.get('l', 0))
            lang_map = global_cfg.get('parsed_language_map', {})
            lang_name = lang_map.get(lang_id, f"lang_{lang_id}")
            msg = str(task.get('msg'))

            log_dir = os.path.join(global_cfg.get('output_dir', ''), "logs", lang_name, msg)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"phase1_build_{datetime.now().strftime('%Y%m%d')}.log")

            futures[msg] = executor.submit(
                run_phase1_pipeline, task, global_cfg, asrmlg_exp_dir,
                python_exec, train_script, log_file
            )

        for msg, future in futures.items():
            status = "SUCCESS" if future.result() else "FAILED"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: phase1_{msg}")


# =============================================================================
# phase 2: incremental testset generation
# =============================================================================

def phase2_frontend_worker(file_path: str, history_hash: Any, task: dict, global_cfg: dict, log_file: str, temp_dir: str, lang_name: str, output_base_dir: str) -> Tuple[str, bool, str, Optional[str], bool]:
    """execute text extraction and semantic hash calculation."""
    current_hash = DeltaTracker.get_semantic_hash(file_path)
    history_hash_str = history_hash.get("hash") if isinstance(history_hash, dict) else history_hash

    # Check if output zip exists
    excel_basename = os.path.splitext(os.path.basename(file_path))[0]
    zip_path = os.path.join(output_base_dir, "test_sets", lang_name, f"{excel_basename}.zip")
    output_exists = os.path.exists(zip_path)

    if current_hash == history_hash_str and output_exists:
        return file_path, False, current_hash, None, True

    hash_val = current_hash[:8]
    txt_temp_path = os.path.join(temp_dir, f"{hash_val}.txt")

    python_exec = global_cfg.get('python_exec', 'python')
    adapter_script = global_cfg.get('adapter_script', '')
    
    adapter_cmd = [
        python_exec, adapter_script,
        "-i", file_path, "-o", txt_temp_path, "-n", "1000"
    ]
    
    success = run_subprocess(adapter_cmd, global_cfg.get('tools_dir', ''), log_file)
    return file_path, True, current_hash, txt_temp_path, success


def phase2_backend_serial(file_path: str, txt_temp_path: str, task: dict, global_cfg: dict, lang: str, log_file: str) -> bool:
    """execute tts synthesis and package the testset."""
    python_exec = global_cfg.get('python_exec', 'python')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir', '')
    test_script = global_cfg.get('test_script')
    output_base_dir = global_cfg.get('output_dir', '')

    if not test_script:
        test_script = os.path.join(global_cfg.get('tools_dir', 'tools'), 'make_test_set.py')

    excel_basename = os.path.splitext(os.path.basename(file_path))[0]
    testset_output_dir = os.path.join(output_base_dir, "test_sets", lang, excel_basename)

    compiler_cmd = [
        python_exec, test_script,
        "-e", asrmlg_exp_dir,
        "-l", str(task.get('l')),
        "-i", txt_temp_path,
        "--output", testset_output_dir,
        "--tts"
    ]
    if task.get('input_wav'):
        compiler_cmd.extend(["-iw", str(task.get('input_wav'))])

    success = run_subprocess(compiler_cmd, asrmlg_exp_dir, log_file)

    if os.path.exists(txt_temp_path):
        try:
            os.remove(txt_temp_path)
        except OSError:
            pass

    return success


def execute_testset_phase(tasks: List[dict], global_cfg: dict):
    """orchestrate phase 2 execution with parallel extraction and serial tts."""
    print("\n=== pipeline phase 2: parallel extraction + serial tts ===")

    output_base_dir = global_cfg.get('output_dir', '')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir', '')

    run_uuid = uuid.uuid4().hex[:8]
    run_temp_dir = os.path.join(output_base_dir, "test_sets", f"temp_{run_uuid}")
    os.makedirs(run_temp_dir, exist_ok=True)

    try:
        for task in tasks:
            if not task.get('enable_testset'):
                continue

            msg = str(task.get('msg'))
            lang_id = str(task.get('l', 0))
            lang_map = global_cfg.get('parsed_language_map', {})
            lang_name = lang_map.get(lang_id, f"lang_{lang_id}")
            corpus_dir = os.path.join(asrmlg_exp_dir, task.get('excel_corpus_path', ''))

            if not os.path.exists(corpus_dir):
                continue

            manifest_path = os.path.join(output_base_dir, "test_sets", f"{lang_name}_{msg}_testset_manifest.json")
            tracker = DeltaTracker(manifest_path)
            log_dir = os.path.join(output_base_dir, "logs", lang_name, msg)

            excel_files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.endswith('.xlsx') and not f.startswith('~')]

            print(f"[{datetime.now().strftime('%H:%M:%S')}] starting parallel hash & text extraction...")
            frontend_results = []
            
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = []
                for file_path in excel_files:
                    file_key = os.path.basename(file_path)
                    history_hash = tracker.history.get(file_key)
                    log_file = os.path.join(log_dir, f"testset_{os.path.splitext(file_key)[0]}_{datetime.now().strftime('%Y%m%d')}.log")
                    
                    futures.append(executor.submit(
                        phase2_frontend_worker, file_path, history_hash, task, global_cfg, log_file, run_temp_dir, lang_name, output_base_dir
                    ))
                
                for future in as_completed(futures):
                    frontend_results.append(future.result())

            print(f"[{datetime.now().strftime('%H:%M:%S')}] starting serial tts generation & packaging...")
            
            for file_path, is_modified, current_hash, txt_temp_path, ext_success in frontend_results:
                excel_basename = os.path.basename(file_path)
                
                if not is_modified:
                    print(f"[skip] no changes: {excel_basename}")
                    continue
                    
                if not ext_success or not txt_temp_path:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] failed: {excel_basename} (text extraction failed)")
                    continue

                print(f"[{datetime.now().strftime('%H:%M:%S')}] processing tts: {excel_basename}...")
                log_file = os.path.join(log_dir, f"testset_{os.path.splitext(excel_basename)[0]}_{datetime.now().strftime('%Y%m%d')}.log")
                
                tts_success = phase2_backend_serial(file_path, txt_temp_path, task, global_cfg, lang_name, log_file)

                if tts_success:
                    tracker.update_history(file_path, current_hash)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] success: {excel_basename}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] failed: {excel_basename} (tts generation failed)")

            tracker.save()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] task completed: manifest saved to {manifest_path}")
            
    finally:
        if os.path.exists(run_temp_dir):
            shutil.rmtree(run_temp_dir, ignore_errors=True)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] cleanup: removed isolated temporary directory {run_temp_dir}")


# =============================================================================
# phase 3: baseline evaluation
# =============================================================================

def execute_eval_phase(tasks: List[dict], global_cfg: dict, executor: ProcessPoolExecutor):
    """execute phase 3 baseline evaluation routing."""
    print("\n=== pipeline phase 3: baseline evaluation ===")

    python_exec = global_cfg.get('python_exec', 'python')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir', '')
    eval_script = os.path.join(
        asrmlg_exp_dir, global_cfg.get('eval_script', 'evaluate.py')
    )

    futures = {}
    for task in tasks:
        if not task.get('enable_eval'):
            continue

        msg = str(task.get('msg'))
        log_dir = os.path.join(global_cfg.get('output_dir', ''), "logs", msg)
        log_file = os.path.join(log_dir, f"phase3_eval_{datetime.now().strftime('%Y%m%d')}.log")

        cmd = [python_exec, eval_script, "--msg", msg]

        print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: eval_{msg}")
        futures[msg] = executor.submit(run_subprocess, cmd, asrmlg_exp_dir, log_file)

    for msg, future in futures.items():
        status = "SUCCESS" if future.result() else "FAILED"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: eval_{msg}")


# =============================================================================
# main entry
# =============================================================================

def main():
    """main pipeline application entry point."""
    parser = argparse.ArgumentParser(description="asr pipeline executor")
    parser.add_argument('-g', '--global_config', required=True, help='path to global_config.yaml')
    parser.add_argument('-j', '--job', required=True, help='path to job.yaml')
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))

    with open(args.global_config, 'r', encoding='utf-8') as f:
        global_cfg = yaml.safe_load(f)
    with open(args.job, 'r', encoding='utf-8') as f:
        job_cfg = yaml.safe_load(f)

    global_cfg = resolve_and_bind_paths(global_cfg, base_path)
    tasks = job_cfg.get('tasks', [])

    lang_map_path = global_cfg.get('language_map_name', '')
    global_cfg['parsed_language_map'] = load_language_map(
        os.path.join(global_cfg.get('asrmlg_exp_dir', ''), lang_map_path)
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: phase 1 and phase 2 concurrently...")

    with ThreadPoolExecutor(max_workers=2) as macro_executor:
        future_p1 = macro_executor.submit(execute_phase1, tasks, global_cfg)
        future_p2 = macro_executor.submit(execute_testset_phase, tasks, global_cfg)
        
        future_p1.result()
        future_p2.result()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] COMPLETED: phase 1 and phase 2. moving to phase 3.")

    with ProcessPoolExecutor(max_workers=4) as executor:
        execute_eval_phase(tasks, global_cfg, executor)


if __name__ == "__main__":
    main()