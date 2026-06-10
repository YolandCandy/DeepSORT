import os
import glob
import argparse

def filter_tracking_results(input_dir, output_dir, conf_thresh):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
    total_removed = 0
    total_kept = 0
    
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        out_file = os.path.join(output_dir, filename)
        
        kept_lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    try:
                        conf = float(parts[6])
                        if conf >= conf_thresh:
                            kept_lines.append(line)
                            total_kept += 1
                        else:
                            total_removed += 1
                    except ValueError:
                        kept_lines.append(line)
                        total_kept += 1
                else:
                    kept_lines.append(line)
                    total_kept += 1
                    
        with open(out_file, 'w') as f:
            f.writelines(kept_lines)
            
    print(f"Processed {len(txt_files)} files.")
    print(f"Kept {total_kept} boxes.")
    print(f"Removed {total_removed} trash boxes (confidence < {conf_thresh}).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter MOT tracking files by confidence')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing .txt files')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save filtered .txt files')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold (e.g. 0.5)')
    args = parser.parse_args()
    
    filter_tracking_results(args.input, args.output, args.thresh)
