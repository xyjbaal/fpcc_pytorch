# Dependencies

torch (1.7.0)


# train

- Move the train data such as "ObjectA_train_91.h5" to the file "datas/ring_train/"

- Rewrite the file list in ring_train.txt.

- run: python fpcc_train.py


# test
- run: python fpcc_test.py --model_file=epoch_25.pt, 

The epoch_25.pt is the model save in "ring_vdm_asm/" 