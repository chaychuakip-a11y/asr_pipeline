"""
ASR Pipeline Executor

ASR 自动化流水线执行器，支持以下功能：
- Phase 1: 资源打包与 G2P (发音预测)
- Phase 2: 增量测试集生成
- Phase 3: 模型评估 (TODO)

Usage:
    python pipeline_executor.py -g config/global_config.yaml -j config/job.yaml
"""
import uuid
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import pandas as pd
import yaml

# =============================================================================
# 工具函数 (Utility Functions)
# =============================================================================

def load_language_map(file_path: str) -> dict:
    """加载语言映射文件。"""
    if not os.path.exists(file_path):
        print(f"Error: Language map file not found at {file_path}", file=sys.stderr)
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


def run_subprocess(cmd: list, cwd: str, log_file: str) -> bool:
    """执行子进程命令并记录日志。"""
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] CMD: {' '.join(cmd)}\n")
            f.flush()
            process = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
            return process.returncode == 0
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\n[FATAL ERROR] {str(e)}\n")
        return False


def resolve_and_bind_paths(global_cfg: dict, base_path: str) -> dict:
    """解析并绑定配置中的相对路径为绝对路径。"""
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


def safe_copy_oov_content(src_path: str, dst_path: str, log_file: str) -> bool:
    """安全拷贝 OOV 文件到 G2P 输入端（严格保留原始二进制编码与换行符）。"""
    try:
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[ERROR] Failed to copy OOV content from {src_path} to {dst_path}: {str(e)}\n")
        return False


# =============================================================================
# 增量追踪器 (Delta Tracker)
# =============================================================================

class DeltaTracker:
    """基于语义哈希的增量追踪器，用于避免重复处理未变更的文件。"""

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
        """
        静态方法：计算文件的语义哈希值。
        
        支持业务模板与普通语料降级：
        - 识别 'sent'/'shuofa' 列或 '<>' 标签页
        - 降级处理：无特殊页时读取第一列或 'text' 列
        - 失败时回退到 MD5 文件哈希
        """
        try:
            # 引入在进程池中可能需要的局部依赖，确保进程安全
            import pandas as pd
            import hashlib
            
            xl = pd.ExcelFile(file_path)
            content_list = []
            has_special_sheet = False

            for sheet in xl.sheet_names:
                sheet_lower = sheet.lower()
                
                # 提取固定句子或说法模板
                if 'sent' in sheet_lower or 'shuofa' in sheet_lower:
                    has_special_sheet = True
                    df = pd.read_excel(xl, sheet_name=sheet, header=None)
                    vals = df.values.flatten()
                    content_list.extend([
                        str(x).strip() for x in vals
                        if pd.notna(x) and str(x).strip()
                    ])
                    
                # 提取槽位字典
                elif '<>' in sheet:
                    has_special_sheet = True
                    df = pd.read_excel(xl, sheet_name=sheet)
                    for col in df.columns:
                        col_name = str(col).strip()
                        if col_name.startswith('<') and col_name.endswith('>'):
                            vals = df[col].dropna().astype(str).str.strip().tolist()
                            content_list.extend([col_name] + vals)

            # 降级：如果没有找到特殊模板页，则作为普通语料处理
            if not has_special_sheet:
                df = pd.read_excel(xl, sheet_name=0)
                if 'text' in df.columns:
                    vals = df['text'].dropna().astype(str).str.strip().tolist()
                else:
                    vals = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                content_list.extend(vals)

            if not content_list:
                raise ValueError("No valid text content found.")

            # 将提取出的所有有效文本拼接，计算哈希
            content_str = "|".join(content_list)
            return hashlib.md5(content_str.encode('utf-8')).hexdigest()

        except Exception as e:
            import hashlib
            # 终极降级：如果 Excel 解析失败或文件格式异常，直接计算文件的物理 MD5
            # print(f"[Warning] Semantic hash failed for {os.path.basename(file_path)}: {str(e)}") # 可选：打印警告日志
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

    def save(self):
        """批量落盘：将内存中的更新一次性写入 JSON。"""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
            
    def update_history(self, file_path: str, new_hash: str):
        """直接更新指定文件的哈希记录，保持与 warmup 一致的字典结构。"""
        from datetime import datetime
        file_key = os.path.basename(file_path)
        
        self.history[file_key] = {
            "hash": new_hash,
            "processed_time": datetime.now().strftime("%Y%m%d_%H%M%S")
        }


# =============================================================================
# Phase 1: 资源打包与 G2P
# =============================================================================

def build_base_command(task: dict, python_exec: str, train_script: str,
                       asrmlg_exp_dir: str) -> list:
    """构建基础命令行参数。"""
    base_cmd = [
        python_exec if train_script.endswith('.py') else "bash",
        train_script
    ]

    if 'l' in task:
        base_cmd.extend(["-l", str(task['l'])])
    if 'G' in task:
        base_cmd.extend(["-G", str(task['G'])])
    if 'is_yun' in task:
        base_cmd.extend(["--is_yun", str(task['is_yun'])])
    if 'excel_corpus_path' in task:
        corpus_abs_path = os.path.join(asrmlg_exp_dir, task['excel_corpus_path'])
        base_cmd.extend(["--excel_corpus_path", corpus_abs_path])
    if 'msg' in task:
        base_cmd.extend(["--msg", str(task['msg'])])
    if task.get('predict_phone_for_new'):
        base_cmd.append("--predict_phone_for_new")

    return base_cmd


def step1_extract_oov(base_cmd: list, task_out_path: str, msg: str,
                      asrmlg_exp_dir: str, log_file: str) -> bool:
    """Phase 1 Step 1: 提取 OOV 词汇。"""
    task_out_path_temp = task_out_path + "_temp"
    cmd = base_cmd + ["--only_corpus_process", "--output", task_out_path_temp]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: Phase1_Step1 (Extract OOV) for {msg}")
    return run_subprocess(cmd, asrmlg_exp_dir, log_file)


def step2_g2p_predict(task: dict, global_cfg: dict, msg: str, task_out_path: str,
                      log_file: str) -> bool:
    """Phase 1 Step 2: G2P 发音预测 (进程排他锁模式，对抗底层硬编码冲突)。"""

    lang_abbr_map = global_cfg.get('lang_abbr_map', {})
    task_out_path_temp = task_out_path + "_temp"
    oov_file_path = os.path.join(
        task_out_path_temp, "custom_corpus_process", "dict_dir", "aaa_oov_base_dict"
    )

    if not os.path.exists(oov_file_path) or os.path.getsize(oov_file_path) == 0:
        return True

    lang_id = str(task.get('l', ''))
    lang_abbr = lang_abbr_map.get(lang_id) or str(task.get('language', msg))

    g2p_root = global_cfg.get('g2p_root_dir')
    g2p_lang_dir = Path(g2p_root) / str(lang_abbr) / "g2p_models"

    if not g2p_lang_dir.is_dir():
        warning_msg = f"\n[WARNING] G2P directory not found: {g2p_lang_dir}. Skipping.\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(warning_msg)
        return True

    # 底层脚本共享的固定输入输出路径
    g2p_input_txt = g2p_lang_dir / "input.txt"
    g2p_output_dict_shared = g2p_lang_dir / "output.dict"
    
    # 我们自己的私有提取路径，防止产物被后续任务冲刷掉
    private_output_dict = Path(task_out_path_temp) / f"g2p_output_{msg}.dict"

    # 嗅探编码 (智能兼容 UTF-16 和 UTF-8)
    target_encoding = 'utf-8'
    target_newline = '\n'
    if g2p_input_txt.exists():
        with open(g2p_input_txt, 'rb') as f_probe:
            raw_bytes = f_probe.read(2)
            if raw_bytes in (b'\xff\xfe', b'\xfe\xff'):
                target_encoding = 'utf-16'
                target_newline = '\r\n'

    print(f"[{datetime.now().strftime('%H:%M:%S')}] WAITING LOCK: Phase1_Step2 for {msg}")

    # ==========================================
    # 临界区：进程排他锁 (防 frontinfo.txt 冲突)
    # ==========================================
    lock_file_path = g2p_lang_dir / ".g2p_engine_exec.lock"
    
    try:
        with open(lock_file_path, 'w') as lock_file:
            # 申请排他锁，拿不到锁的并发任务会在这里乖乖排队，不消耗 CPU
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ACQUIRED LOCK: Executing G2P for {msg}")
                
                # 1. 写入 OOV 文本
                with open(oov_file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                    content = f_in.read()
                with open(g2p_input_txt, 'w', encoding=target_encoding, newline=target_newline) as f_out:
                    f_out.write(content)
                    
                # 2. 执行危险的底层黑盒脚本
                g2p_script_cmd = ["bash", "run.sh"]
                success = run_subprocess(g2p_script_cmd, str(g2p_lang_dir), log_file)
                
                # 3. 火速将战利品私有化，防止被下一秒进来的任务覆盖
                if success and g2p_output_dict_shared.exists():
                    shutil.copy2(g2p_output_dict_shared, private_output_dict)
                else:
                    return False
            finally:
                # 无论发生什么天崩地裂的错误，绝对保证锁被释放，拯救系统
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] RELEASED LOCK: G2P finished for {msg}")
                
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f_log:
            f_log.write(f"[ERROR] Locked G2P execution failed: {str(e)}\n")
        return False

    return True

def step3_merge_dict(task: dict, global_cfg: dict, msg: str, log_file: str) -> bool:
    """Phase 1 Step 3: 合并预测发音到主词典。"""
    # 动态获取资源目录映射
    res_dir_map = global_cfg.get('res_dir_map', {})
    lang_abbr_map = global_cfg.get('lang_abbr_map', {})

    merge_script = global_cfg.get('merge_dict_script')
    res_base_dir = os.path.join(global_cfg.get('asrmlg_exp_dir'), global_cfg.get('res_dir_name'))

    if not (merge_script and os.path.exists(merge_script) and res_base_dir):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[WARNING] merge_dict_script or res_root_dir missing. Skipping Merge.\n")
        return True

    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: Phase1_Step3 (Merge OOV Dict) for {msg}")

    python_exec = global_cfg.get('python_exec', 'python')
    vcs_script_path = os.path.join(global_cfg.get('tools_dir', './'), 'lexicon_vcs.py')

    lang_id = str(task.get('l', 0))
    lang_abbr = lang_abbr_map.get(lang_id)

    is_yun_val = str(task.get('is_yun', '0'))
    # 动态解析资源子目录
    res_dir_name = res_dir_map.get(is_yun_val, "unknown")

    lang_map = global_cfg.get('parsed_language_map', {})
    lang_name = lang_map.get(lang_id, "")

    target_res_dir = os.path.join(res_base_dir, f"{lang_name}_res", res_dir_name)
    target_dict_path = os.path.join(target_res_dir, "new_dict")
    phone_syms_path = os.path.join(target_res_dir, "phones.syms")
    g2p_output_dict = os.path.join(
        global_cfg.get('g2p_root_dir', ''), lang_abbr, "g2p_models", "output.dict"
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] merging {target_dict_path}")

    if not os.path.exists(g2p_output_dict):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[ERROR] g2p_output_dict missing at {g2p_output_dict}. Aborting Merge.\n")
        return False

    # Pre-merge
    if os.path.exists(vcs_script_path):
        run_subprocess([python_exec, vcs_script_path, "-i", target_dict_path, "pre_merge"],
                       global_cfg.get('asrmlg_exp_dir'), log_file)

    # 合并
    merge_cmd = [
        python_exec, merge_script,
        "-i", g2p_output_dict,
        "-o", target_dict_path,
        "-p", phone_syms_path
    ]
    if task.get('predict_phone_for_new'):
        merge_cmd.append("--predict_phone_for_new")

    merge_success = run_subprocess(merge_cmd, global_cfg.get('asrmlg_exp_dir'), log_file)

    # Post-merge
    if merge_success and os.path.exists(vcs_script_path):
        run_subprocess([
            python_exec, vcs_script_path,
            "-i", target_dict_path,
            "post_merge",
            "-m", task.get('msg', ''),
            "-l", lang_id,
            "--max_versions", str(global_cfg.get('max_versions', 10))
        ], global_cfg.get('asrmlg_exp_dir'), log_file)

    return True


def step4_full_build(base_cmd: list, task_out_path: str, msg: str,
                     patch_type: str, asrmlg_exp_dir: str, log_file: str) -> bool:
    """Phase 1 Step 4: 全量资源打包。"""
    # 清理临时目录
    task_out_path_temp = task_out_path + "_temp"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CLEANING: temp dir")
    try:
        shutil.rmtree(task_out_path_temp)
        print(f"Successfully deleted: {task_out_path_temp}")
    except OSError as e:
        print(f"Error: {task_out_path_temp} : {e.strerror}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: Phase1_Step4 (Full Resource Build) for {msg}")

    base_cmd = base_cmd + ["--output", task_out_path]
    try:
        idx = base_cmd.index("--msg")
        base_cmd[idx + 1] = f"{msg}_{patch_type}"
    except (ValueError, IndexError):
        base_cmd.extend(["--msg", f"{msg}_{patch_type}"])

    return run_subprocess(base_cmd, asrmlg_exp_dir, log_file)


def run_phase1_pipeline(task: dict, global_cfg: dict, asrmlg_exp_dir: str,
                        python_exec: str, train_script: str, log_file: str) -> bool:
    """
    执行 Phase 1 完整流水线。

    Steps:
        1. 提取 OOV
        2. G2P 预测 (可选)
        3. 合并词典 (可选)
        4. 全量资源打包
    """
    msg = str(task.get('msg'))
    lang_id = str(task.get('l', ''))
    base_out_dir = global_cfg.get('output_dir')
    resolved_msg = str(task.get('msg', '')).strip()
    scheme_map = global_cfg.get('scheme_map', {})

    lang_map = global_cfg.get('parsed_language_map', {})
    lang_name = lang_map.get(lang_id, f"lang_{lang_id}")
    model_type = scheme_map.get(str(task.get('is_yun', 0)), 'unknown')
    task_out_path = os.path.join(
        base_out_dir, lang_name, resolved_msg,
        f"{model_type}_{datetime.now().strftime('%Y%m%d')}"
    )

    # 构建基础命令
    base_cmd = build_base_command(task, python_exec, train_script, asrmlg_exp_dir)

    # Step 1: 提取 OOV
    if not step1_extract_oov(base_cmd, task_out_path, msg, asrmlg_exp_dir, log_file):
        return False

    # Step 2: G2P 预测
    if task.get('enable_g2p'):
        if not step2_g2p_predict(task, global_cfg, msg, task_out_path, log_file):
            return False

    # Step 3: 合并词典
    if task.get('enable_merge_dict'):
        step3_merge_dict(task, global_cfg, msg, log_file)

    # Step 4: 全量资源打包
    is_yun_val = str(task.get('is_yun', '0'))
    patch_type = scheme_map.get(is_yun_val, "unknown")
    return step4_full_build(base_cmd, task_out_path, msg, patch_type, asrmlg_exp_dir, log_file)


def execute_phase1(tasks: list, global_cfg: dict):
    """执行 Phase 1 调度 (内部自动进行多进程并发)。"""
    print("\n=== Pipeline Phase 1: Resource Build & G2P (Parallel Mode) ===")

    python_exec = global_cfg.get('python_exec')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir')
    train_script = os.path.join(
        asrmlg_exp_dir, global_cfg.get('train_script', 'corpus_process_package.py')
    )

    futures = {}
    # 把进程池的创建移到函数内部
    with ProcessPoolExecutor(max_workers=4) as executor:
        for task in tasks:
            if not task.get('enable_g2p') and not task.get('enable_merge_dict'):
                continue

            lang_id = str(task.get('l', 0))
            lang_map = global_cfg.get('parsed_language_map', {})
            lang_name = lang_map.get(lang_id, f"lang_{lang_id}")
            msg = str(task.get('msg'))

            log_dir = os.path.join(global_cfg.get('output_dir'), "logs", lang_name, msg)
            log_file = os.path.join(log_dir, f"phase1_build_{datetime.now().strftime('%Y%m%d')}.log")

            futures[msg] = executor.submit(
                run_phase1_pipeline, task, global_cfg, asrmlg_exp_dir,
                python_exec, train_script, log_file
            )

        for msg, future in futures.items():
            status = "SUCCESS" if future.result() else "FAILED"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: Phase1_{msg}")


# =============================================================================
# Phase 2: 测试集生成
# =============================================================================
def phase2_frontend_worker(file_path: str, history_hash: str, task: dict, global_cfg: dict, log_file: str, temp_dir: str):
    """
    前端并行任务：计算哈希并抽取文本。
    """
    current_hash = DeltaTracker.get_semantic_hash(file_path)
    if current_hash == history_hash:
        return file_path, False, current_hash, None, True

    python_exec = global_cfg.get('python_exec')
    adapter_script = global_cfg.get('adapter_script')
    
    hash_val = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:8]
    # 使用传入的独立沙箱目录进行无锁写入
    txt_temp_path = os.path.join(temp_dir, f"{hash_val}.txt")
    
    adapter_cmd = [
        python_exec, adapter_script,
        "-i", file_path, "-o", txt_temp_path, "-n", "1000"
    ]
    
    success = run_subprocess(adapter_cmd, global_cfg.get('tools_dir'), log_file)
    return file_path, True, current_hash, txt_temp_path, success

def phase2_backend_serial(file_path: str, txt_temp_path: str, task: dict, global_cfg: dict, lang: str, log_file: str) -> bool:
    """
    后端串行任务：使用 TTS 生成语音并打包。
    """
    python_exec = global_cfg.get('python_exec')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir')
    test_script = global_cfg.get('test_script')
    output_base_dir = global_cfg.get('output_dir')

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

    # 修正错误：清理临时生成的 txt，而不是清理包含有效产物的 testset_output_dir
    if os.path.exists(txt_temp_path):
        try:
            os.remove(txt_temp_path)
        except OSError as e:
            print(f"Cleanup warning: {e.strerror}")

    return success

def execute_testset_phase(tasks: list, global_cfg: dict):
    """执行 phase 2 测试集生成 (架构：前端高并发抽取 -> 后端严格串行 tts)。"""
    print("\n=== pipeline phase 2: parallel extraction + serial tts ===")

    output_base_dir = global_cfg.get('output_dir')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir')

    # 为当前流水线任务创建完全隔离的专属 temp 目录
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
                    
                    # 挂载专属的 run_temp_dir 到 worker
                    futures.append(executor.submit(
                        phase2_frontend_worker, file_path, history_hash, task, global_cfg, log_file, run_temp_dir
                    ))
                
                for future in as_completed(futures):
                    frontend_results.append(future.result())

            print(f"[{datetime.now().strftime('%H:%M:%S')}] starting serial tts generation & packaging...")
            
            for file_path, is_modified, current_hash, txt_temp_path, ext_success in frontend_results:
                excel_basename = os.path.basename(file_path)
                
                if not is_modified:
                    print(f"[skip] no changes: {excel_basename}")
                    continue
                    
                if not ext_success:
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
        # 兜底操作：强制抹除本周期的专属资源栈，保证环境清洁且互不干扰
        if os.path.exists(run_temp_dir):
            shutil.rmtree(run_temp_dir, ignore_errors=True)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] cleanup: removed isolated temporary directory {run_temp_dir}")

# =============================================================================
# Phase 3: 模型评估
# =============================================================================

def execute_eval_phase(tasks: list, global_cfg: dict, executor: ProcessPoolExecutor):
    """执行 Phase 3 模型评估 (TODO)。"""
    print("\n=== Pipeline Phase 3: Baseline Evaluation ===")

    python_exec = global_cfg.get('python_exec')
    asrmlg_exp_dir = global_cfg.get('asrmlg_exp_dir')
    eval_script = os.path.join(
        asrmlg_exp_dir, global_cfg.get('eval_script', 'evaluate.py')
    )

    futures = {}
    for task in tasks:
        if not task.get('enable_eval'):
            continue

        msg = str(task.get('msg'))
        log_dir = os.path.join(global_cfg.get('output_dir'), "logs", msg)
        log_file = os.path.join(log_dir, f"phase3_eval_{datetime.now().strftime('%Y%m%d')}.log")

        cmd = [python_exec, eval_script, "--msg", msg]

        print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: Eval_{msg}")
        futures[msg] = executor.submit(run_subprocess, cmd, asrmlg_exp_dir, log_file)

    for msg, future in futures.items():
        status = "SUCCESS" if future.result() else "FAILED"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: Eval_{msg}")


# =============================================================================
# 主入口 (Main Entry)
# =============================================================================

def main():
    """主函数入口。"""
    parser = argparse.ArgumentParser(description="ASR Pipeline Executor")
    parser.add_argument('-g', '--global_config', required=True, help='Path to global_config.yaml')
    parser.add_argument('-j', '--job', required=True, help='Path to job.yaml')
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))

    # 加载配置
    with open(args.global_config, 'r', encoding='utf-8') as f:
        global_cfg = yaml.safe_load(f)
    with open(args.job, 'r', encoding='utf-8') as f:
        job_cfg = yaml.safe_load(f)

    global_cfg = resolve_and_bind_paths(global_cfg, base_path)
    tasks = job_cfg.get('tasks', [])

    lang_map_path = global_cfg.get('language_map_name')
    global_cfg['parsed_language_map'] = load_language_map(
        os.path.join(global_cfg.get('asrmlg_exp_dir'), lang_map_path)
    )

    # =========================================================================
    # 宏观流水线调度 (Macro Pipeline Orchestration)
    # =========================================================================
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: Phase 1 and Phase 2 Concurrently...")

    # 使用多线程调度器同时启动 Phase 1 和 Phase 2 
    # (因为它们内部各自有多进程调度，所以外层用轻量级线程即可，避免创建进程池中嵌套进程池)
    with ThreadPoolExecutor(max_workers=2) as macro_executor:
        future_p1 = macro_executor.submit(execute_phase1, tasks, global_cfg)
        future_p2 = macro_executor.submit(execute_testset_phase, tasks, global_cfg)
        
        # 强制主程序阻塞等待，直到 Phase 1 和 Phase 2 都全部彻底完成
        future_p1.result()
        future_p2.result()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] COMPLETED: Phase 1 and Phase 2. Moving to Phase 3.")

    # =========================================================================
    # Phase 3: 模型评估 (依赖 Phase 1 和 Phase 2 的产出)
    # =========================================================================
    # 在 1 和 2 完全结束后，安全地启动 Phase 3
    with ProcessPoolExecutor(max_workers=4) as executor:
        execute_eval_phase(tasks, global_cfg, executor)


if __name__ == "__main__":
    main()