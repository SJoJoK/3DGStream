import os
import shutil

base_paths = ["./Code-Release/flame_steak"]

for base_path in base_paths:
    frame_folders = sorted([f for f in os.listdir(base_path) if f.startswith('frame') and os.path.isdir(os.path.join(base_path, f))])

    ntcs_path = os.path.join(base_path, 'NTCs')
    addition_3dgs_path = os.path.join(base_path, 'additional_3dgs')
    pre_frame_3dgs_path = os.path.join(base_path, 'pre-frame_3dgs')
    raw_path = os.path.join(base_path, 'raw') 
    os.makedirs(ntcs_path, exist_ok=True)
    os.makedirs(addition_3dgs_path, exist_ok=True)
    os.makedirs(pre_frame_3dgs_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)

    for folder in frame_folders:
        frame_id = int(folder[-6:])

        ntc_path = os.path.join(base_path, folder, 'NTC.pth')
        if os.path.isfile(ntc_path):
            ntc_target_path = os.path.join(ntcs_path, f'NTC_{frame_id-1:06}.pth')
            shutil.copy(ntc_path, ntc_target_path)

        addition_3dgs_source_path = os.path.join(base_path, folder, 'point_cloud', 'iteration_250', 'added', 'point_cloud.ply')
        if os.path.isfile(addition_3dgs_source_path):
            addition_3dgs_target_path = os.path.join(addition_3dgs_path, f'additions_{frame_id-1:06}.ply')
            shutil.copy(addition_3dgs_source_path, addition_3dgs_target_path)

        shutil.move(os.path.join(base_path, folder), raw_path)

    print(f"Files in {base_path} have been reorganized.")
