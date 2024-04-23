import os
import shutil
import argparse

def copy_sparse_to_frames(source, scene):
    sparse_dir = os.path.join(source, 'sparse')
    if not os.path.isdir(sparse_dir):
        print(f"Error: The directory '{sparse_dir}' does not exist.")
        return

    for item in os.listdir(scene):
        frame_dir = os.path.join(scene, item)
        if os.path.isdir(frame_dir) and item.startswith('frame'):
            dest_sparse_dir = os.path.join(frame_dir, 'sparse')
            if os.path.exists(dest_sparse_dir):
                shutil.rmtree(dest_sparse_dir)
            shutil.copytree(sparse_dir, dest_sparse_dir)
            print(f"Copied to {dest_sparse_dir}")

def copy_distorted_to_scene(source, scene):
    distorted_dir = os.path.join(source, 'distorted')
    if not os.path.isdir(distorted_dir):
        print(f"Error: The directory '{distorted_dir}' does not exist.")
        return

    dest_distorted_dir = os.path.join(scene, 'distorted')
    if os.path.exists(dest_distorted_dir):
        shutil.rmtree(dest_distorted_dir)
    shutil.copytree(distorted_dir, dest_distorted_dir)
    print(f"Copied to {dest_distorted_dir}")

def main(args):
    copy_sparse_to_frames(args.source, args.scene)
    copy_distorted_to_scene(args.source, args.scene)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy directories to specified locations.')
    parser.add_argument('--source', type=str, help='The source directory containing sparse and distorted folders.')
    parser.add_argument('--scene', type=str, help='The scene directory.')
    
    args = parser.parse_args()
    main(args)
