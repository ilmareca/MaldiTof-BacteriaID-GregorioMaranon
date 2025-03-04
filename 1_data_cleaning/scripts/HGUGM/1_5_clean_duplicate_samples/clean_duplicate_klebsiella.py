import os
import shutil

def clean_duplicate_samples(base_dir):
    for extern_id in os.listdir(base_dir):
        extern_id_dir = os.path.join(base_dir, extern_id)
        if os.path.isdir(extern_id_dir):
            target_positions = os.listdir(extern_id_dir)
            
            # Keep only one target position
            if len(target_positions) > 1:
                for target_position in target_positions[1:]:
                    target_position_dir = os.path.join(extern_id_dir, target_position)
                    if os.path.isdir(target_position_dir):
                        shutil.rmtree(target_position_dir)
            
            # Ensure only one folder named '1' exists in the target position
            target_position_dir = os.path.join(extern_id_dir, target_positions[0])
            if os.path.isdir(target_position_dir):
                subdirs = os.listdir(target_position_dir)
                for subdir in subdirs:
                    subdir_path = os.path.join(target_position_dir, subdir)
                    if os.path.isdir(subdir_path) and subdir != '1':
                        shutil.rmtree(subdir_path)

if __name__ == "__main__":
    base_dir = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/klebsiellaPneumoniae_v2'
    clean_duplicate_samples(base_dir)