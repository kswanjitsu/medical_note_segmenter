# Medical Note Segmentation Using Deep Learning

This project aims to create a deep learning approach to segment medical notes into subjective, objective, and assessment+plan sections. We use the self-instruct method adapted from the Stanford Alpaca paper.

## Data

We collected 100 notes from MTsamples, divided them into 1-10 subsections, and labeled them as 'subjective', 'objective', 'assessment_plan.' This resulted in around 1,000 gold standard labels that were used as both seed tasks and as training examples. In total, we had over 4,000 examples split into train, validation, and test sets.

The model base is a SciFive (T5) Med NLI that was then fine-tuned to perform classification. On our held-out test set, the model achieves accuracy, precision, and recall of 100%.

## Files

- `01_train_set.csv`: The training dataset
- `02_val_set.csv`: The evaluation dataset
- `03_test_set.csv`: The held-out test set, never used in training

All of these files were created by first generating the synthetic data from text davinci 003 with the seed tasks and then splitting them into their respective subsets.

- `prompt.txt`: The prompt used for text davinci to generate synthetic examples. This was modified from the original self-instruct and alpaca repo.
- `seed_tasks.jsonl`: The seed tasks created from the 100 MTsamples.
- `generate_instructions.py`: A file taken from the Alpaca GitHub to generate the synthetic data from the seed set.
- `utils.py`: Utility functions from the Alpaca repo used in the `generate_instructions.py` script.
- `train_t5_segmenter.ipynb` : The training and validation script, we used SimpleT5 a wrapper over pytorch lightning.

## Model Weights
These are too large to store in this repo.
[You can access the model weights here](https://huggingface.co/kswanjitsu/medical_note_segmenter)

## Results
On a hold out test test of 80+ examples not seen during training or for evaluation we achieve 100% accuracy, recall and precision.
You can look at these results in the jupyter notebook file towards the end.
