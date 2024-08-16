import matplotlib.pyplot as plt


name = 'BraTS_T1,T1ce,T2_flair_finetune'

# 로그 파일 경로
log_file_path = f'./checkpoints/{name}/log.txt'

# 에포크별 mean_psnr을 저장할 리스트
epochs = []
mean_psnr_values = []

# 로그 파일 읽기
with open(log_file_path, 'r') as file:
    for line in file:
        if 'Epoch' in line and 'mean_psnr' in line:
            parts = line.split()
            epoch = int(parts[1])
            mean_psnr = float(parts[-3])
            # mean_psnr = float(parts[6][-5:])
            epochs.append(epoch)
            mean_psnr_values.append(mean_psnr)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_psnr_values, marker='o', linestyle='-', color='b')
plt.title(f'{name}')
plt.xlabel('Epoch')
plt.ylabel('Mean PSNR')
plt.grid(True)
# plt.show()

plt.savefig(f'./checkpoints/{name}/mean_psnr_over_epochs.png')

