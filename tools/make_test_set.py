# -*- coding:utf-8 -*-
import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path
import zipfile
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

def run_tts_generation(input_txt: str, language: int, output_dir: str):
    # [修改点] 动态绑定 XTTS 引擎路径
    tts_base_dir = ENGINE_DIR / "xtts20_for_asr" / "bin_tts"
    wav_outdir = tts_base_dir / "wav_outdir"
    frontinfo = tts_base_dir / "frontinfo.txt"
    
    if frontinfo.exists():
        frontinfo.unlink()
    if wav_outdir.exists():
        shutil.rmtree(wav_outdir)
    wav_outdir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env['TTSKNL_DOMAIN'] = str(tts_base_dir)
    env['LD_LIBRARY_PATH'] = str(tts_base_dir)
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
    
    log_file = tts_base_dir / "xtts_predict.log"
    print(f"[INFO] Running TTS Generation. Log: {log_file}")
    
    with open(log_file, 'w', encoding='utf-8') as f_log:
        try:
            subprocess.run(cmd, cwd=tts_base_dir, env=env, stdout=f_log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[FATAL] XTTS Engine failed with return code {e.returncode}. Check {log_file}")
            sys.exit(1)

    final_output_txt = Path(output_dir) / f"{Path(output_dir).name}.txt"
    
    #获取 wav_outdir 下所有生成的音频文件名，并按数字顺序排序
    wav_files = sorted([f.name for f in wav_outdir.iterdir() if f.is_file()])
    
    if not wav_files:
        print(f"[ERROR] No wav files generated by TTS in {wav_outdir}")
        sys.exit(1)

    #将生成的文件名与输入文本按行拼接，中间用 Tab 隔开
    with open(input_abs_path, 'r', encoding='utf-8') as f_in, \
         open(final_output_txt, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        for wav_name, line in zip(wav_files, lines):
            f_out.write(f"{wav_name}\t{line.strip()}\n")
    
    if final_output_txt.exists():
        return str(final_output_txt), str(wav_outdir)
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
        
        with open(args.txt_path, 'r', encoding='utf-8') as infile, \
             open(filter_txt, 'w', encoding='utf-8') as outfile, \
             open(oov_filter_txt, 'w', encoding='utf-8') as outfile_oov:
             
            for line in infile:
                line = line.strip()
                if not line: continue
                corpus_proc_inst.filter_corpus_by_char(line, outfile, outfile_oov, args.post)
                
        output_path, gen_wav_dir = run_tts_generation(str(filter_txt), args.language, args.output)
        build_testset_package(output_path, args.language, gen_wav_dir, args.output, args.post, corpus_proc_inst)
        
    else:
        build_testset_package(args.txt_path, args.language, args.input_wav, args.output, args.post, corpus_proc_inst)

if __name__ == "__main__":
    main()