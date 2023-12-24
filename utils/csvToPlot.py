import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('../Loss_results/Loss_NN_DZ_residual_train20.csv', index_col=None)
epoch_column = df1['epoch']
test_loss_abs_column = df1['test_loss_abs']
test_loss_RMS_column = df1['test_loss_RMS']


plt.plot(epoch_column, test_loss_RMS_column, marker='o')


df2 = pd.read_csv('../Loss_results/Loss_NN_DZ_residual_train20_batch491.csv', index_col=None)
epoch_column = df2['epoch']
test_loss_abs_column = df2['test_loss_abs']
test_loss_RMS_column = df2['test_loss_RMS']

plt.plot(epoch_column, test_loss_RMS_column, marker='*')

plt.title('Loss Function Plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['20', '491'])
plt.grid(True)
# plt.savefig('../Loss_results/NN_WS4_residual_train20_test.png')
plt.show()