# shot-clustering

Tested on Python 3.5 with OpenCV 3.0.0 (+contrib)

To run the main algorithm:
```bash
python -m sceneprocessor.main --help
usage: main.py [-h] [--verbose] --input-dir INPUT_DIR
               [--output-file OUTPUT_FILE] [--plot]
               [--frame-limit FRAME_LIMIT]

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Display informational output
  --input-dir INPUT_DIR
                        Path to input sequence directory
  --output-file OUTPUT_FILE
                        Path to output text file
  --plot                Plot metrics
  --frame-limit FRAME_LIMIT
                        Number of frames to limit the sequence to
```