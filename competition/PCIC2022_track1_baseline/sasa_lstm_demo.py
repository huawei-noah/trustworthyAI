import os
from sasa_lstm.sasa_lstm_ import sasa_lstm_model

data_path = "./dataset/"
model = sasa_lstm_model(data_path=data_path)
model.train()
res = model.predict()

if not os.path.exists('result'):
    os.makedirs('result')
res.to_csv("result/submission.csv", index=False, header=False)
