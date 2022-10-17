# coding = utf-8

import os
from sasa.estimator import SASA_Classiffer


if __name__ == '__main__':
    # data path
    src_train_path = os.path.join("Dataset/phase1_TrainData/cityA/X.npy")
    tgt_train_path = os.path.join("Dataset/phase1_TrainData/cityB/train/X.npy")
    tgt_test_path = os.path.join("Dataset/phase1_TestData/X.npy")

    clsf = SASA_Classiffer(input_dim=10, training_steps=100)
    clsf.fit(src_train_path, tgt_train_path)
    clsf.predict(tgt_test_path)
