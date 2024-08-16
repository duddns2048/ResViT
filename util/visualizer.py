import numpy as np
import os
import ntpath
import time
from . import util
from . import html
# from scipy.misc import imresize
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt
import math

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id # display_id=0 |  window id of the web display  뭐임이거
        self.use_html = opt.isTrain
        self.win_size = opt.display_winsize # display_winsize = 256 | display window size 뭐임이거 
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.error_history = {'G_GAN': [], 'G_L1': [], 'D_real': [], 'D_fake': []}
        self.epochs = []
        self.val_epochs=[]
        self.psnr_history = []
        self.ssim_history = []
        
        if self.display_id > 0:
            # python -m visdom.server | 터미널에 로컬 서버열고 시작하기
            # http://localhost:8097 # port num으로 이동
            import visdom # visualize 객체: 로컬서버, 포트 지정
            self.vis = visdom.Visdom(server=opt.display_server, # "http://localhost"
                                     port=opt.display_port) 

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir]) # 리스트이면 리스트안의 경로 dir 생성, 그냥 str이면 그거만 생성
            
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file: # a: append(추가하기) 모드
            now = time.strftime("%c") # str format time: '현재시간'을 '문자열'로 반환. '%c': 'Tue May 14 08:19:07 2024'
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals):
        if self.display_id > 0:  # show images in the browser | window id of the web display 이게 뭐임
            # visdom에 realA, fakeB, realB 생성
            idx = 1
            for label, image_numpy in visuals.items(): # visuals: dict(realA, fakeB, realB)
                if image_numpy.shape[-1] != 1:
                    tmp = np.concatenate([image_numpy[:,:,i] for i in range(image_numpy.shape[-1])],axis=1)
                    self.vis.image(tmp, opts=dict(title=label),
                                win=self.display_id + idx)
                else:
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                win=self.display_id + idx)
                idx += 1

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, error_dict, dataset_size=1):
        if not hasattr(self, 'plot_data'): # hasattr(a,b): a객체에 self.b 가 있는지 True/False 출력
            self.plot_data = {'X': [], 'Y': [], 'legend': list(error_dict.keys())}
        self.plot_data['X'].append(epoch) # epoch
        self.plot_data['Y'].append([np.mean(error_dict[k]) for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
    
    def plot_current_errors_png(self, epoch, error_dict):
        self.epochs.append(epoch)
        for key in error_dict:
            self.error_history[key].append(np.mean(error_dict[key]))

        plt.figure(figsize=(10, 5))
        for key, values in self.error_history.items():
            plt.plot(self.epochs, values, label=key)

        # 그래프 제목 및 축 레이블 설정
        plt.title(f'Error: {self.name}')
        plt.xlabel('Epoch')
        plt.ylabel('Error Values')

        plt.legend()
        plt.grid(True)
        plt.savefig(f'./checkpoints/{self.opt.name}/error_plot_epoch.png')
    
    def plot_current_psnr_png(self, psnr_avg):
        self.psnr_history.append(psnr_avg)

        # 플롯 생성
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_epochs, self.psnr_history, label='PSNR')

        plt.title(f'PSNR {self.opt.name}')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR Values')

        plt.legend()
        plt.grid(True)
        plt.savefig(f'./checkpoints/{self.opt.name}/PSNR_plot_epoch.png')
        
    def plot_current_ssim_png(self, ssim_avg):
        self.ssim_history.append(ssim_avg)

        plt.figure(figsize=(10, 5))
        plt.plot(self.val_epochs, self.ssim_history, label='SSIM')

        plt.title(f'SSIM {self.opt.name}')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM Values')

        plt.legend()
        plt.grid(True)
        plt.savefig(f'./checkpoints/{self.opt.name}/SSIM_plot_epoch.png')
        
    def plot_current_metric_png(self, epoch, psnr_avg, ssim_avg):
        self.val_epochs.append(epoch)
        self.plot_current_psnr_png(psnr_avg)
        self.plot_current_ssim_png(ssim_avg)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, error_dict, dataset_size, current_iter, write):
        message = f'epoch: {epoch} | iter: [{(current_iter)}/{dataset_size}]'
        for k, v in error_dict.items():
            if write:
                message += f'{k}: {np.mean(v)} '
            else:
                message += f'{k}: {v} '

        print(message)
        if write:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, aspect_ratio=1.0):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, im in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = resize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = resize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
