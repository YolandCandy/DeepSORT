import os
import glob
import argparse

def compute_iob(box, region):
    # box and region: [left, top, width, height]
    b_left, b_top, b_w, b_h = box
    r_left, r_top, r_w, r_h = region
    
    b_right = b_left + b_w
    b_bottom = b_top + b_h
    r_right = r_left + r_w
    r_bottom = r_top + r_h
    
    i_left = max(b_left, r_left)
    i_top = max(b_top, r_top)
    i_right = min(b_right, r_right)
    i_bottom = min(b_bottom, r_bottom)
    
    if i_right <= i_left or i_bottom <= i_top:
        return 0.0
        
    i_area = (i_right - i_left) * (i_bottom - i_top)
    b_area = b_w * b_h
    
    if b_area == 0:
        return 0.0
        
    return i_area / b_area

def filter_by_ignore_regions(trk_dir, ignore_base_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    trk_files = glob.glob(os.path.join(trk_dir, '*.txt'))
    total_removed = 0
    total_kept = 0
    
    for trk_file in trk_files:
        filename = os.path.basename(trk_file)
        seq_name = os.path.splitext(filename)[0]
        
        # Load ignore regions
        ignore_file = os.path.join(ignore_base_dir, seq_name, 'ignored_regions.txt')
        ignore_regions = []
        if os.path.exists(ignore_file):
            with open(ignore_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        ignore_regions.append([float(x) for x in parts[:4]])
                        
        out_file = os.path.join(out_dir, filename)
        kept_lines = []
        
        with open(trk_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        box = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                        
                        # Check overlap with any ignore region
                        is_ignored = False
                        for region in ignore_regions:
                            if compute_iob(box, region) > 0.5:
                                is_ignored = True
                                break
                                
                        if not is_ignored:
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
            
    print(f"Processed {len(trk_files)} sequences.")
    print(f"Kept {total_kept} boxes.")
    print(f"Removed {total_removed} boxes overlapping with ignore regions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trk', required=True)
    parser.add_argument('--ignore', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    filter_by_ignore_regions(args.trk, args.ignore, args.out)
