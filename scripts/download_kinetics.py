import os
import uuid
import shutil
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


def create_video_folders(annotations, save_dir):
    """Creates a directory for each label name in the dataset"""
    label_names = [annot[0] for annot in annotations]
    label_names = set(label_names)

    for label_name in label_names:
        dir_name = save_dir / label_name
        if not dir_name.exists():
            dir_name.mkdir(parents=True, exist_ok=True)
    return label_names


def download_clip(annot, save_dir, tmp_dir):
    num_attempts = 5
    url_base = 'https://www.youtube.com/watch?v='
    status = False
    label, youtube_id, start, end = annot
    assert len(youtube_id) == 11
    filename = save_dir / label / f"{youtube_id}_{start}_{end}.mp4"
    tmp_filename = tmp_dir / f"{uuid.uuid4()}.mp4"

    command = ['yt-dlp', '--quiet', '--no-warnings', '-f', 'mp4', '-o', f'"{tmp_filename}"', f'"{url_base + youtube_id}"']
    command = ' '.join(command)

    for i in range(5):
        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            if i == num_attempts - 1:
                return youtube_id, status, err.output.decode("utf-8")
        else:
            break

    command1 = ['ffmpeg', '-i', f'"{str(tmp_filename)}"', '-ss', str(start), '-t', str(end - start), '-c:v', 'libx264', '-c:a', 'copy', '-threads', '1', '-loglevel', 'panic', f'"{str(filename)}"']
    command1 = ' '.join(command1)

    try:
        output = subprocess.check_output(command1, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return youtube_id, status, err.output.decode("utf-8")

    status = filename.exists()
    os.remove(tmp_filename)
    return youtube_id, status, 'Downloaded'


def main(csv_path, save_dir, num_jobs, classes=None):
    split = csv_path.split('.')[0].split('_')[-1]
    save_dir = Path(save_dir)
    save_dir = save_dir / split
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(save_dir)
    
    tmp_dir = save_dir / 'tmp'
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
    with open(csv_path) as f:
        lines = f.read().splitlines()[1:]
    
    print(f"Total video files for all classes: {len(lines)}")

    annotations = []
    for line in lines:
        label, youtube_id, start, end, _, _ = line.split(",")
        label = label.strip('"')

        if classes is not None:
            if label not in classes:
                continue

        if " " in label:
            words = label.split(" ")
            label = "_".join(words)

        annotations.append((label, youtube_id, int(start), int(end)))

    print(f"Total video files to download: {len(annotations)}")
    
    class_names = create_video_folders(annotations, save_dir)

    print(f"Number of classes to download: {len(class_names)} {class_names}")

    if num_jobs == 1:
        pass
    else:
        status_lst = Parallel(num_jobs)(delayed(download_clip)(annot, save_dir, tmp_dir) for annot in tqdm(annotations))

    shutil.rmtree(tmp_dir)

    reports = ["youtube_id,log\n"]
    failed = 0

    for id, status, log in status_lst:
        if not status:
            failed += 1
            reports.append(f"{id},{log}")

    report_path = save_dir / "download_report.csv"
    with open(save_dir, 'w') as f:
        f.writelines(reports)

    print(f"Videos downloaded: {len(annotations)-failed}")
    print(f"Videos cannot be downloaded: {failed}")
    print(f"See the log at {str(report_path)}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='data/kinetics-400_val.csv', help="kinetics csv path")
    parser.add_argument("--save_dir", type=str, default='data/k400', help="download dir")
    parser.add_argument("--num_jobs", type=int, default=24, help="num of jobs to download")
    parser.add_argument("--classes", type=list, default=None, help="limit the classes to download")
    args = parser.parse_args()

    main(**vars(args))