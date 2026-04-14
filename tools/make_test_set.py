# -*- coding:utf-8 -*-
import sys
import os
import shutil
import argparse
import subprocess
import zipfile
import uuid       # [新增] 用于生成独立任务 ID
import fcntl      # [新增] 用于进程排他锁
from pathlib import Path

# ==========================================
# 动态路径注入 (Dynamic Path Injection)
# ==========================================
# 提前解析 -e/--engine_dir 参数，确保能在 import 之前挂载主工程的环境
_init_parser = argparse.ArgumentParser(add_help=False)
_init_parser.add_argument('-e', '--engine_dir', type=str, required=True, help="Absolute path to main ASR engine directory")
_init_args, _ = _init_parser.parse_known_args()

ENGINE_DIR = Path(_init_args.engine_dir).resolve()
PYTHON_LIB_PATH = ENGINE_DIR / "python_lib"

if not PYTHON_LIB_PATH.exists():
    print(f"[FATAL ERROR] python_lib not found in {ENGINE_DIR}. Please check --engine_dir.")
    sys.exit(1)

# 将主工程的 python_lib 强行压入环境变量最前端
sys.path.insert(0, str(ENGINE_DIR))
if PYTHON_LIB_PATH.exists():
    sys.path.insert(0, str(PYTHON_LIB_PATH))

# 现在可以安全地导入主工程的内部库了
from corpus_process import *

def generate_mlf(input_txt_path: str, output_mlf_path: str):
    with open(input_txt_path, 'r', encoding='utf-8') as src_file, \
         open(output_mlf_path, 'w', encoding='utf-8') as dst_mlf:
        
        dst_mlf.write("#!MLF!#\n")
        for line in src_file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            wave_name_full = parts[0]
            lab_line = parts[-1].strip()
            wave_name = Path(wave_name_full).stem
            
            dst_mlf.write(f'"*/{wave_name}.lab"\n')
            dst_mlf.write("<s>\n")
            for word in lab_line.split(' '):
                if word.strip():
                    dst_mlf.write(f"{word}\n")
            dst_mlf.write("</s>\n.\n")

def process_text_corpus(input_txt: str, language: int, ispost: bool, corpus_proc_inst) -> str:
    input_path = Path(input_txt)
    output_txt = input_path
    
    if num2LagDict[language] in need_split:
        before_split = input_path.with_name(f"{input_path.name}_before_split")
        shutil.move(input_path, before_split)
        split_function = corpus_process.get_split_function()
        
        with open(before_split, 'r', encoding='utf-8') as infile, \
             open(output_txt, 'w', encoding='utf-8') as outfile, \
             open(input_path.with_name(f"{input_path.name}_split_oov"), 'w', encoding='utf-8') as oov_file:
            for line in infile:
                line = line.strip()
                if not line: continue
                parts = line.split("\t")
                wave_name = parts[0]
                lab_line = parts[-1].strip()
                outfile.write(f"{wave_name}\t")
                split_function.split(lab_line, outfile, oov_file)

    before_filter = input_path.with_name(f"{input_path.name}_before_filter")
    shutil.move(output_txt, before_filter)
    
    with open(before_filter, 'r', encoding='utf-8') as infile, \
         open(output_txt, 'w', encoding='utf-8') as outfile, \
         open(input_path.with_name(f"{input_path.name}_filter_oov"), 'w', encoding='utf-8') as oov_file:
        for line in infile:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            wave_name = parts[0]
            lab_line = parts[-1].strip()
            outfile.write(f"{wave_name}\t")
            corpus_proc_inst.filter_corpus_by_char(lab_line, outfile, oov_file, ispost)
            
    return str(output_txt)

def run_tts_generation(input_txt: str, language: int, output_dir: str, label_txt: str = None):
    """
    [修改点] 引入 fcntl 进程排他锁，彻底解决多任务并发时抢夺引擎产生的冲突。
    """
    tts_base_dir = ENGINE_DIR / "xtts20_for_asr" / "bin_tts"
    wav_outdir = tts_base_dir / "wav_outdir"
    frontinfo = tts_base_dir / "frontinfo.txt"
    
    # 任务专属的绝对安全输出目录 (防并发冲刷)
    task_uuid = uuid.uuid4().hex[:6]
    private_wav_dir = Path(output_dir) / f"wavs_{task_uuid}"
    log_file = Path(output_dir) / f"xtts_predict_{task_uuid}.log"
    
    env = os.environ.copy()
    env['TTSKNL_DOMAIN'] = str(tts_base_dir)
    # [修改点] 安全追加系统路径，防止覆盖导致引擎缺失核心库
    original_ld = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f"{tts_base_dir}:{original_ld}" if original_ld else str(tts_base_dir)
    env['OMP_NUM_THREADS'] = '1'
    env['XTTS_VERSION'] = 'Travis'
    
    input_abs_path = Path(input_txt).resolve()
    lang_param = ttsdict[num2LagDict[language]]
    
    cmd = [
        "./ttsSample", 
        "-l", "libttsknl.so", 
        "-v", str(lang_param), 
        "-x", "1", 
        "-i", str(input_abs_path), 
        "-o", "wav_outdir/", 
        "-m", "1", "-f", "1", "-g", "1"
    ]
    
    # [修改点] 全局排他锁
    lock_file = tts_base_dir / ".tts_global_engine.lock"
    print(f"[INFO] Task {task_uuid} waiting for TTS Engine Lock...")

    with open(lock_file, 'w') as lf:
        # 获取排他锁，阻塞其他并发任务
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            print(f"[INFO] Acquired Lock! Running TTS Generation for Task {task_uuid}. Log: {log_file}")
            
            # 清理共享的战场
            if frontinfo.exists():
                frontinfo.unlink()
            if wav_outdir.exists():
                shutil.rmtree(wav_outdir)
            wav_outdir.mkdir(parents=True, exist_ok=True)
            
            # 运行霸道的底层引擎
            with open(log_file, 'w', encoding='utf-8') as f_log:
                subprocess.run(cmd, cwd=tts_base_dir, env=env, stdout=f_log, stderr=subprocess.STDOUT, check=True)
            
            # [关键修改] 趁着锁还没释放，赶紧把生成的音频拷贝到专属的安全目录下！
            if private_wav_dir.exists():
                shutil.rmtree(private_wav_dir)
            shutil.copytree(wav_outdir, private_wav_dir)
            
        except subprocess.CalledProcessError as e:
            print(f"[FATAL] XTTS Engine failed with return code {e.returncode}. Check {log_file}")
            sys.exit(1)
        finally:
            # 释放锁，让下一个任务进入
            fcntl.flock(lf, fcntl.LOCK_UN)
            print(f"[INFO] Lock Released by Task {task_uuid}")

    # ==========================================
    # 以下逻辑在锁外执行 (不阻塞其他任务)
    # ==========================================
    final_output_txt = Path(output_dir) / f"{Path(output_dir).name}.txt"
    
    # [修改点] 从私有的安全目录读取音频文件列表
    wav_files = sorted([f.name for f in private_wav_dir.iterdir() if f.is_file()])
    
    if not wav_files:
        print(f"[ERROR] No wav files generated by TTS in {private_wav_dir}")
        sys.exit(1)

    # Use label_txt if provided, otherwise use input_txt
    label_source = Path(label_txt).resolve() if label_txt else input_abs_path
    with open(label_source, 'r', encoding='utf-8') as f_in, \
         open(final_output_txt, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        if len(wav_files) != len(lines):
            print(f"[WARNING] Mismatch between wav files ({len(wav_files)}) and labels ({len(lines)})")
        
        for wav_name, line in zip(wav_files, lines):
            f_out.write(f"{wav_name}\t{line.strip()}\n")
    
    if final_output_txt.exists():
        # [修改点] 返回私有目录，后续的 Zip 打包将使用这个安全目录
        return str(final_output_txt), str(private_wav_dir)
    else:
        print("[ERROR] wav_label_file.txt generation failed.")
        sys.exit(1)

def build_testset_package(input_txt: str, language: int, input_wav_dir: str, output_dir: str, ispost: bool, corpus_proc_inst):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    target_txt = out_path / f"{out_path.name}.txt"
    target_mlf = out_path / f"{out_path.name}.mlf"
    
    if not Path(input_txt).exists():
        print(f"[ERROR] Source text not found: {input_txt}")
        return

    # 1. 同步文本
    if Path(input_txt).resolve() != target_txt.resolve():
        shutil.copy2(input_txt, target_txt)
    
    # 2. 文本清洗与 MLF 生成
    processed_txt = process_text_corpus(str(target_txt), language, ispost, corpus_proc_inst)
    generate_mlf(processed_txt, str(target_mlf))
    
    # 3. 精准打包：音频 + MLF (去除任何中间层级)
    if input_wav_dir and Path(input_wav_dir).exists():
        archive_path = out_path.parent / f"{out_path.name}.zip"
        zip_root_name = out_path.name  # Zip 内的顶层文件夹名
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # A. 写入 .mlf 文件
            if target_mlf.exists():
                zipf.write(target_mlf, arcname=f"{zip_root_name}/{target_mlf.name}")
                
            # B. 遍历原始音频目录，直接打入 Zip 的顶层文件夹下
            wav_src_path = Path(input_wav_dir)
            for file_path in wav_src_path.iterdir():
                if file_path.is_file():
                    zip_internal_path = f"{zip_root_name}/{file_path.name}"
                    zipf.write(file_path, arcname=zip_internal_path)
                        
        print(f"[INFO] Testset packaged successfully (Strict Flat Mode): {archive_path}")

def main():
    parser = argparse.ArgumentParser(description="Make Test Set Pipeline")
    parser.add_argument('-e', '--engine_dir', type=str, required=True, help="Absolute path to main ASR engine directory")
    
    parser.add_argument('-l', '--language', type=int, required=True)
    parser.add_argument('-i', '--txt_path', type=str, required=True, help="Input text path")
    parser.add_argument('--output', type=str, required=True, help="Output directory path")
    parser.add_argument('-iw', '--input_wav', type=str, default=None, help="Real wav directory path")
    parser.add_argument('--tts', action='store_true', help="Enable TTS generation")
    parser.add_argument('--post', action='store_true', help="Post-processing flag")
    parser.add_argument('--replacement_list', type=str, default=None, help="Path to replacement list file")
    
    args = parser.parse_args()

    out_dir = Path(args.output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    corpus_proc_inst = get_corpus_process(
        args.language, None, None, None, None, None, None, None, None, None, None, None
    )

    if args.tts:
        filter_txt = out_dir / "filter.txt"
        oov_filter_txt = out_dir / "oov_filter.txt"
        synthesize_txt = out_dir / "synthesize.txt"
        oov_synthesize_txt = out_dir / "oov_synthesize.txt"

        # Load replacement rules BEFORE any filtering
        # Replacements must be applied to the original text so keys can match unmodified words.
        print(f"[DEBUG] Checking replacement list: {args.replacement_list}")
        replacements = {}
        if args.replacement_list and os.path.exists(args.replacement_list):
            with open(args.replacement_list, 'r', encoding='utf-8') as f_rep:
                for rep_line in f_rep:
                    rep_line = rep_line.strip()
                    if not rep_line or ':' not in rep_line:
                        continue
                    parts = rep_line.split(':', 1)
                    if len(parts) == 2:
                        replacements[parts[0].strip().upper()] = parts[1].strip()
            print(f"[DEBUG] Loaded {len(replacements)} replacement rules (Case-Insensitive).")
        else:
            print(f"[DEBUG] Replacement list not found or not provided. Path: {args.replacement_list}")

        # Single pass over raw input:
        #   - filter original line  → filter.txt   (label / ground-truth)
        #   - apply replacements to original, then filter → synthesize.txt (TTS input)
        # Doing both in one pass guarantees 1-to-1 line correspondence before sync.
        match_count = 0
        with open(args.txt_path, 'r', encoding='utf-8') as infile, \
             open(filter_txt, 'w', encoding='utf-8') as f_filter, \
             open(oov_filter_txt, 'w', encoding='utf-8') as f_filter_oov, \
             open(synthesize_txt, 'w', encoding='utf-8') as f_synth, \
             open(oov_synthesize_txt, 'w', encoding='utf-8') as f_synth_oov:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                # Build replacement text from the ORIGINAL (pre-filter) words
                words = line.split()
                new_words = []
                for w in words:
                    upper_w = w.upper()
                    if upper_w in replacements:
                        new_words.append(replacements[upper_w])
                        match_count += 1
                    else:
                        new_words.append(w)
                replaced_line = " ".join(new_words)

                # Filter original → label
                corpus_proc_inst.filter_corpus_by_char(line, f_filter, f_filter_oov, args.post)
                # Filter replaced → TTS input
                corpus_proc_inst.filter_corpus_by_char(replaced_line, f_synth, f_synth_oov, args.post)

        print(f"[DEBUG] Applied {match_count} replacements.")

        # Sync filter.txt and synthesize.txt: drop any line-pair where either side is empty.
        # This prevents TTS from skipping blank lines and creating a wav-count mismatch.
        with open(filter_txt, 'r', encoding='utf-8') as fa, \
             open(synthesize_txt, 'r', encoding='utf-8') as fb:
            filter_lines = fa.readlines()
            synth_lines = fb.readlines()

        synced_filter, synced_synth = [], []
        for fl, sl in zip(filter_lines, synth_lines):
            if fl.strip() and sl.strip():
                synced_filter.append(fl if fl.endswith('\n') else fl + '\n')
                synced_synth.append(sl if sl.endswith('\n') else sl + '\n')

        dropped = len(filter_lines) - len(synced_filter)
        if dropped:
            print(f"[WARNING] Dropped {dropped} line(s) where filter produced empty output.")

        with open(filter_txt, 'w', encoding='utf-8') as fa, \
             open(synthesize_txt, 'w', encoding='utf-8') as fb:
            fa.writelines(synced_filter)
            fb.writelines(synced_synth)

        print(f"[DEBUG] Synthesis text prepared: {synthesize_txt.resolve()} ({len(synced_synth)} lines)")

        output_path, gen_wav_dir = run_tts_generation(str(synthesize_txt), args.language, args.output, label_txt=str(filter_txt))
        build_testset_package(output_path, args.language, gen_wav_dir, args.output, args.post, corpus_proc_inst)
        
    else:
        build_testset_package(args.txt_path, args.language, args.input_wav, args.output, args.post, corpus_proc_inst)

if __name__ == "__main__":
    main()