# A Simple yet Effective Self-Debiasing Framework for Transformer Models
The repository is modified from https://github.com/UKPLab/emnlp2020-debiasing-unknown.

## Requirements
The code requires python >= 3.6 and pytorch >= 1.1.0.

Additional required dependencies can be found in `requirements.txt`.
Install all requirements by running:
```bash
pip install -r requirements.txt
```

## Data
All datasets we used in this paper are attached. Please unpack it under the current directory.
```
unzip datasets.zip
```

## Running Experiments
We use the the `--dataset` arguments to set the appropriate dataset, which can be set as `mnli/fever/QQP`. To start training our framework, run the following:
```
OUTPUT_DIR=/path/to/save/checkpoints
DATASET=fever

python train_distill_bert.py --output_dir $OUTPUT_DIR --do_eval --mode none --dataset $DATASET --num_train_epochs 5
```