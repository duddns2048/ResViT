import matplotlib.pyplot as plt
import re
import os
# 데이터 파일 경로
dir_path = './checkpoints'
task_name = 'BraTS_T1,T1ce,T2_flair_finetune'
file_name = 'loss_log.txt'

file_path = os.path.join(dir_path, task_name, file_name)

# 데이터를 저장할 리스트 초기화
epochs = []
G_GAN_values = []
G_L1_values = []
D_real_values = []
D_fake_values = []

# 데이터 파일 읽기
with open(file_path, 'r') as file:
    for line in file:
        if '(epoch:' in line and 'G_GAN:' in line:
            parts = line.split()
            epoch = int(parts[1][:-1])  # '(epoch:' 부분 제거
            G_GAN = float(parts[5])
            G_L1 = float(parts[7])
            D_real = float(parts[9])
            D_fake = float(parts[11])
            
            epochs.append(epoch)
            G_GAN_values.append(G_GAN)
            G_L1_values.append(G_L1)
            D_real_values.append(D_real)
            D_fake_values.append(D_fake)

# 데이터 플로팅
plt.figure(figsize=(12, 8))
plt.plot(epochs, G_GAN_values, label='G_GAN')
plt.plot(epochs, G_L1_values, label='G_L1')
plt.plot(epochs, D_real_values, label='D_real')
plt.plot(epochs, D_fake_values, label='D_fake')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{task_name} Losses over Epochs')
plt.legend()
plt.grid(True)

output_path = os.path.join(dir_path, task_name, task_name + 'losses_over_epochs.png')
plt.savefig(output_path)

plt.show()