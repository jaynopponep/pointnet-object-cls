## Notes 
- So far capable of running on CPU slowly and get results (tested 25 epochs so far)
- To add more data, download the data with .PCD data in it, and add it to trainingdata/

## General Workflow or Setting Up
1. `python3 -m venv venv`
2. `pip install -r requirements.txt`
3. Run the PCD -> HDF5 (Python binaries) converter with `python3 PCD_to_HDF5.py` 
4. Train: `python3 train.py`
4a. Note, if you are trying to swap from CPU -> GPU, you will have to interact with `CUDA_VISIBLE_DEVICES`, and the argument `--gpu`, and possibly line 105ish that looks for the tf.device() for training. 
5. After training, should be able to test and classify: `python3 test.py` and `python3 classify.py`. This should generate a confusion matrix on the CLI for you
6. Profit
