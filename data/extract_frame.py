import argparse
import os
import cv2
import glob

def extract_frames(src_dir, out_dir, frame_rate):
    """Extract frames from videos at a specific frame rate."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    video_files = sorted(glob.glob(os.path.join(src_dir, '*.mp4')))  # Assumes videos are .mp4
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        base_name = os.path.basename(video_file).split('.')[0]
        video_out_dir = os.path.join(out_dir, base_name)
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)

        # Check video frame rate
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps != 25:
            print(f"Warning: Video '{video_file}' has a frame rate of {actual_fps} FPS, not 25 FPS.")

        count = 0
        skip_frames = int(actual_fps / frame_rate)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % skip_frames == 0:
                # Resize frame to 451x256
                resized_frame = cv2.resize(frame, (451, 256), interpolation=cv2.INTER_LANCZOS4)
                out_path = os.path.join(video_out_dir, f'img_{count:08d}.jpg')
                cv2.imwrite(out_path, resized_frame)
            count += 1
        cap.release()
    print(f"Frames extracted to {out_dir}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument('--src_dir', type=str, required=True, help="Source directory of videos.")
    parser.add_argument('--out_dir', type=str, required=True, help="Output directory for frames.")
    parser.add_argument('--frame_rate', type=float, default=25, help="Frame rate to extract frames.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.src_dir, args.out_dir, args.frame_rate)

