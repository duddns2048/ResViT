import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to parse the log file
def parse_log(file_path):
    # Regular expression to match the log lines
    log_pattern = re.compile(
        r'\(epoch: (\d+), iters: \d+, time: [\d\.]+, data: [\d\.]+\) G_GAN: ([\d\.]+) G_L1: ([\d\.]+) D_real: ([\d\.]+) D_fake: ([\d\.]+)'
    )
    
    data = {
        'epoch': [],
        'G_GAN': [],
        'G_L1': [],
        'D_real': [],
        'D_fake': []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                epoch, g_gan, g_l1, d_real, d_fake = match.groups()
                data['epoch'].append(int(epoch))
                data['G_GAN'].append(float(g_gan))
                data['G_L1'].append(float(g_l1))
                data['D_real'].append(float(d_real))
                data['D_fake'].append(float(d_fake))
    
    return pd.DataFrame(data)

# Load the data
log_df = parse_log('./checkpoints/BraTS_T1_T2_flair_finetune_00/loss_log.txt')

# Calculate the average loss per epoch
avg_loss_per_epoch = log_df.groupby('epoch').mean()

# Plotting the average loss per epoch
plt.figure(figsize=(12, 8))
plt.plot(avg_loss_per_epoch.index, avg_loss_per_epoch['G_GAN'], label='G_GAN')
plt.plot(avg_loss_per_epoch.index, avg_loss_per_epoch['G_L1'], label='G_L1')
plt.plot(avg_loss_per_epoch.index, avg_loss_per_epoch['D_real'], label='D_real')
plt.plot(avg_loss_per_epoch.index, avg_loss_per_epoch['D_fake'], label='D_fake')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
