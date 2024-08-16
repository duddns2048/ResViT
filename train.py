import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
import os
from util.util import save_image
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
import random

def set_seed(seed):
    # Python의 랜덤 시드 고정
    random.seed(seed)
    # Numpy의 랜덤 시드 고정
    np.random.seed(seed)
    # PyTorch의 시드 고정
    torch.manual_seed(seed)
    # CUDA 연산에서의 시드 고정 (GPU 사용 시)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 여러 개의 GPU를 사용하는 경우 모든 GPU의 시드를 고정
    # CuDNN에서의 재현성 보장 설정 (가능한 경우)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def print_log(logger,message):
    print(f'Project Name: {message}', flush=True)
    if logger:
        logger.write(str(message) + '\n')
        
def compute_psnr(fake_im, real_im):
    if fake_im.shape[0] != real_im.shape[0]:
        raise ValueError('image channel is different!')
    psnr_per_channel = []
    for i in range(fake_im.shape[0]):
        psnr_per_channel.append(psnr(fake_im[i],real_im[i]))
    
    return torch.mean(torch.tensor(psnr_per_channel))

def compute_ssim(fake_im, real_im):
    if fake_im.shape[0] != real_im.shape[0]:
        raise ValueError('image channel is different!')
    ssim_per_channel = []
    for i in range(fake_im.shape[0]):
        ssim_per_channel.append(ssim(fake_im[i].unsqueeze(dim=0),real_im[i].unsqueeze(dim=0)))
    
    return torch.mean(torch.tensor(ssim_per_channel))
##

if __name__ == '__main__':
    set_seed(2048)
    opt = TrainOptions().parse()
    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data() 
    dataset_size = len(dataset)
    print('Number of training images = %d' % dataset_size)

    #validation data
    opt.phase='val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('Number of Validation images = %d' % dataset_size_val)
    
    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name) # name으로 실험 버전 관리
    logger = open(os.path.join(save_dir, 'validation_log.txt'), 'a') # w+:읽기+쓰기 가능 | 없으면 생성 | 있으면 덮어쓰기
    print_log(logger,opt.name)
    logger.close()

    L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])   # loss 리스트: shape : [epoch 수, batch 수]
    psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)]) # loss 리스트: shape : [epoch 수, batch 수]
        
    model = create_model(opt) # opt에 따라 모델 선정, initialize
    visualizer = Visualizer(opt) # visualizer 초기화: 그림저장, log 저장 등을 수행하는 클래스
    
    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda() #because we normalize to 0-1
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    
    lr_list = []
    # lr_list.append(opt.lr)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): # range(1,50+50+1) 총 100번의 epoch인듯 | 고정 lr로 50 epoch | lr decay로 50 epoch
        count=0
        #Training step
        opt.phase='train'
        error_dict = {'G_GAN':[], 'G_L1':[], 'D_real':[], 'D_fake':[]}
        for i, data in tqdm(enumerate(dataset,1)):
            visualizer.reset() # saved 변수를 False로 바꾸는게 끝. 일종의 flag인듯
            model.set_input(data) # data에서 A이미지, B이미지, 경로 꺼내기
            model.optimize_parameters() # forward - D학습 - G학습

            # visdom 서버에 input output 출력
            if i % opt.visdom_upload_freq == 0: # visdom_upload_freq=100 | frequency of showing training results on screen 
                temp_visuals=model.get_current_visuals() # return realA, fakeB, realB in dict form ## 넘파이로 변환(마지막이 C)!!
                visualizer.display_current_results(temp_visuals) # visdom 이미지 저장 관련 함수
                    

            # loss log를 print
            if i % opt.print_freq == 0: # print_freq=100 | frequency of showing training results on console
                errors = model.get_current_errors() # G_GAN, G_L1, D_real, D_fake 를 dict로 반환
                visualizer.print_current_errors(epoch, errors, dataset_size, i, False) # 터미널에 epoch, iters, time, data, 4개의 loss 출력하고 loss_log.txt에 저장

            # epoch 10마다 120step 마다 web.image 폴더에 input, output, gt png로 저장
            if (epoch % 10 ==1):
                temp_visuals=model.get_current_visuals()
                if i <= 10:
                    count+=1
                    for label, image_numpy in temp_visuals.items():
                        img_path = os.path.join(save_dir, 'web/images',f'epoch{epoch}_{count}_{label}.png')
                        if image_numpy.shape[-1] != 1:
                                image_numpy = np.concatenate([image_numpy[:,:,i] for i in range(image_numpy.shape[-1])],axis=1)
                        else:
                            image_numpy = image_numpy[:,:,0]
                        save_image(image_numpy, img_path) # 넘파이를 PIL Image로 바꿔서 저장
               
            current_errors = model.get_current_errors()
            for key in error_dict.keys():
                error_dict[key].append(current_errors[key])

        print('=====================================================================================')
        model.save('latest')
        print(f'saving the latest model (epoch {epoch})')
        
        # 1 epoch 마다 visdom에 loss 그래프 plot
        visualizer.plot_current_errors(epoch, error_dict, dataset_size) # visdom에 그래프 plot
        # 1 epoch 마다 loss 그래프 png 저장
        visualizer.plot_current_errors_png(epoch, error_dict)
        # 1 epoch 마다 터미널에 loss출력, loss_log.txt 저장
        visualizer.print_current_errors(epoch, error_dict, dataset_size, i, True) # 터미널에 loss 출력, loss_log.txt 저장

        

        #Validaiton step
        # 5 epoch 마다 validation 수행
        if epoch % opt.validation_freq == 0: # valdation_freq=5 | frequency of saving checkpoints at the end of epochs
            logger = open(os.path.join(save_dir, 'validation_log.txt'), 'a') # a = 추가모드, 없으면 새로 생성
            opt.phase='val'
            
            psnr_list = []
            ssim_list = []
            
            for i, data_val in enumerate(dataset_val):     		    
                model.set_input(data_val) # data에서 A이미지, B이미지, 경로 꺼내기     		    
                model.test() # with torch.no_grad()하고 realA, fakeB = G(realA), realB 세팅     		    
                if opt.dataset_mode in ['3_to_1', 'uni']:
                    fake_im=model.fake_B[:,model.missing_modality_index:model.missing_modality_index+1,:,:].data
                    real_im=model.real_B[:,model.missing_modality_index:model.missing_modality_index+1,:,:].data
                elif opt.dataset_mode =='3_to_1_real':
                    fake_im=model.fake_B.data
                    real_im=model.real_B.data                    

                fake_im=fake_im*0.5+0.5 	    
                real_im=real_im*0.5+0.5

                L1_avg[epoch-1,i]=abs(fake_im - real_im).mean()
                psnr_list.append(compute_psnr(fake_im, real_im))
                ssim_list.append(compute_ssim(fake_im, real_im))

            l1_avg_loss = np.mean(L1_avg[epoch-1]) 
            # mean_psnr =np.mean(psnr_list[epoch-1])     
            # std_psnr = np.std(psnr_list[epoch-1])     
            
            psnr_avg = torch.mean(torch.tensor(psnr_list))
            ssim_avg = torch.mean(torch.tensor(ssim_list))
            
            visualizer.plot_current_metric_png(epoch, psnr_avg, ssim_avg)
            
            print_log(logger,'Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr: %.3f ' % (epoch, l1_avg_loss, psnr_avg, ssim_avg))             
            logger.close()
    		
            if epoch % 10 == 0:
                print(f'saving the model at the end of epoch {epoch}')
                model.save(epoch)


        current_lr = model.update_learning_rate()
        lr_list.append(current_lr)
        model.plot_lr_graph(lr_list,epoch)
        print('===================================  ==================================================')
