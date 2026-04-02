import os
import sys
import base64
import argparse
from pyzbar.pyzbar import decode
from PIL import Image

def restore_file(image_dir, output_file):
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        sys.exit(1)

    chunks = {}
    total_expected = -1
    processed_count = 0

    # 1. 扫描目录下所有图片 (Scan all images in directory)
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("No images found in the directory.")
        return

    print(f"[*] Scanning {len(files)} files...")

    for f in files:
        try:
            decoded_list = decode(Image.open(f))
            if not decoded_list:
                continue
            
            # 2. 解析 Payload (Parse Payload: index|total|data)
            content = decoded_list[0].data.decode('utf-8')
            parts = content.split('|', 2)
            
            if len(parts) == 3:
                idx = int(parts[0])
                total_expected = int(parts[1])
                data = parts[2]
                
                if idx not in chunks:
                    chunks[idx] = data
                    processed_count += 1
                    
                # 可选：重命名图片方便对齐 (Optional: Auto-rename for tracking)
                new_name = os.path.join(image_dir, f"verified_{idx:03d}.png")
                if f != new_name and not os.path.exists(new_name):
                    os.rename(f, new_name)
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # 3. 完整性检查 (Integrity Check)
    if total_expected == -1:
        print("[-] No valid QR metadata found.")
        return

    print(f"[*] Progress: {len(chunks)}/{total_expected} chunks collected.")

    if len(chunks) < total_expected:
        missing = [i for i in range(total_expected) if i not in chunks]
        print(f"[-] Missing chunks: {missing}")
        print("Please capture the missing QR codes and run again.")
        return

    # 4. 排序并还原文件 (Sort and Reconstruct)
    print("[*] All chunks found. Reconstructing file...")
    full_b64 = "".join([chunks[i] for i in sorted(chunks.keys())])
    
    try:
        with open(output_file, 'wb') as f:
            f.write(base64.b64decode(full_b64))
        print(f"[+] Success! File restored to: {output_file}")
    except Exception as e:
        print(f"[-] Restore failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore file from QR code images.")
    parser.add_argument("dir", help="Directory containing QR code images")
    parser.add_argument("output", help="Output filename (e.g., project_restored.tar.xz)")
    
    args = parser.parse_args()
    restore_file(args.dir, args.output)