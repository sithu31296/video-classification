# Dataset Preparation

## Kinetics-400/600 

Install `ffmpeg` and run the following script to download the videos:

```bash
# Download all classes
$ python scripts/download_kinetics.py \
    --csv_path KINETICS_CSV_PATH
    --save_dir VIDEO_SAVE_DIR
    --num_jobs NUM_PARALLEL_JOBS

# Download specific classes
$ python scripts/download_kinetics.py \
    --csv_path KINETICS_CSV_PATH
    --save_dir VIDEO_SAVE_DIR
    --num_jobs NUM_PARALLEL_JOBS
    --classes ['archery', 'applauding']
```

Kinetics-400 and 600 csv files can be downloaded at [] 
You can see the kinetics class names in [k400_classnames](../data/k400_classnames.txt).