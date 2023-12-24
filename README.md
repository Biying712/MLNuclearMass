# MassPrediction

To use run the code:

1. Install Anaconda
2. `conda create -n MP python=3.8`
3. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
4. `python MassPrediction.py NN
--data=DZresidualfile
--batchsize=64
--epochs=500
--lr=0.1
--gamma=0.9
--save_dir=./model_saved
--num_monte_carlo=20
--results_dir=./results
--test_split=0.8
--mode=train
--log_interval=5
--load_model=
