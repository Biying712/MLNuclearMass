# MassPrediction

To use run the code:

1. Install Anaconda
2. `conda create -n MP python=3.8`
3. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
4. To run: e.g.
`python main.py NN
--data=WS4
--batchsize=2000
--epochs=10000
--lr=0.01
--weight_decay=0.01
--scheduler_gamma=0.95
--scheduler_step=100
--save_dir=./model_saved
--num_monte_carlo=1
--results_dir=./results
--test_split=0.2
--mode=train
--log_interval=1
--load_model= `
