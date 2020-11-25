# COMMANDS TO TRAIN, TEST & PREDICT

## TRAIN:

python main.py --mode="train" --data_dir="<DATA_DIR>" --save_dir="<SAVED_MODEL_DIR>"

## TEST:
python main.py --mode="test" --data_dir="<DATA_DIR>" --save_dir="<SAVED_MODEL_DIR>"

## PREDICT:
python main.py --mode="test" ---test_file"<TEST(npy) file>" --save_dir="<SAVED_MODEL_DIR>"


# NOTES ABOUT CONFIG(Configure.py)

* Set the *version* parameter in Configure.py to choose between the three model versions.
* The three options are v1,v2 & v3.

# Installations required
* Python==3.6
* tensorflow==2.2.0
* numpy==1.16.5
