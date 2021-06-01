import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from natsort import natsorted

first = True
# AdamW_10-40sec_Mel_Spectrogram / AdamW_10-40sec_MFCC/ RAdam_10-40sec_Mel_Spectrogram / RAdam_10-40sec_MFCC/history
PATH = 'AdamW_10-40sec_Mel_Spectrogram'
history_list = natsorted(os.listdir(PATH))
print(history_list)
file_name = history_list[0][:-6]

for idx, name in enumerate(history_list):
    df = pd.read_csv(os.path.join(PATH, name))

    globals()[f'loss_{idx+1}'] = df['loss'].to_list()
    globals()[f'accuracy_{idx+1}'] = df['accuracy'].to_list()
    globals()[f'val_loss_{idx+1}'] = df['val_loss'].to_list()
    globals()[f'val_accuracy_{idx+1}'] = df['val_accuracy'].to_list()

total_loss = list(zip(loss_1, loss_2, loss_3, loss_4, loss_5))
total_acc = list(zip(accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_5))
total_val_loss = list(zip(val_loss_1, val_loss_2, val_loss_3, val_loss_4, val_loss_5))
total_val_acc = list(zip(val_accuracy_1, val_accuracy_2, val_accuracy_3, val_accuracy_4, val_accuracy_5))

loss_lower_bound = [min(x) for x in total_loss]
loss_upper_bound = [max(x) for x in total_loss]
acc_lower_bound = [min(x) for x in total_acc]
acc_upper_bound = [max(x) for x in total_acc]

val_loss_lower_bound = [min(x) for x in total_val_loss]
val_loss_upper_bound = [max(x) for x in total_val_loss]
val_acc_lower_bound = [min(x) for x in total_val_acc]
val_acc_upper_bound = [max(x) for x in total_val_acc]

loss_avg = [sum(x)/len(x) for x in total_loss]
acc_avg = [sum(x)/len(x) for x in total_acc]
val_loss_avg = [sum(x)/len(x) for x in total_val_loss]
val_acc_avg = [sum(x)/len(x) for x in total_val_acc]

index = range(len(acc_avg))
df = pd.DataFrame(data=list(zip(loss_avg, acc_avg, val_loss_avg, val_acc_avg)),
                  columns=['loss', 'accuracy', 'val_loss', 'val_accuracy'], index=index)
df.to_csv(os.path.join(PATH, file_name+'_avg.csv'), mode='w')

x_axis = list(index)
plt.plot(x_axis, acc_avg)
plt.plot(x_axis, val_acc_avg)
plt.fill_between(x_axis, acc_lower_bound, acc_upper_bound, alpha=0.2)
plt.fill_between(x_axis, val_acc_lower_bound, val_acc_upper_bound, alpha=0.2)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.savefig(os.path.join(PATH, file_name+'_avg.png'), dpi=300)
plt.show()


# ============================ Trying to plot LOSS for checking convergence ===================
x_axis = list(index)
plt.plot(x_axis, loss_avg)
plt.plot(x_axis, val_loss_avg)
plt.fill_between(x_axis, loss_lower_bound, loss_upper_bound, alpha=0.2)
plt.fill_between(x_axis, val_loss_lower_bound, val_loss_upper_bound, alpha=0.2)
plt.title('Model accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.savefig(os.path.join(PATH, file_name+'_loss.png'), dpi=300)
plt.show()