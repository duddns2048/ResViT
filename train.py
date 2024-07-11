import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py 
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from util.util import save_image

def print_log(logger,message):
    print(f'Project Name: {message}', flush=True)
    if logger:
        logger.write(str(message) + '\n')
if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Training data
    data_loader = CreateDataLoader(opt) # loader 선언, initialize하여 반환
    dataset = data_loader.load_data() #TODO 그냥 return self : 이거 해야됨? 
    dataset_size = len(data_loader)
    print('Number of training images = %d' % dataset_size)

    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name) # name으로 실험 버전 관리
    logger = open(os.path.join(save_dir, 'log.txt'), 'a') # w+:읽기+쓰기 가능 | 없으면 생성 | 있으면 덮어쓰기
    print_log(logger,opt.name)
    logger.close()

    #validation data
    opt.phase='val'
    data_loader_val = CreateDataLoader(opt) # TODO val도 같은 데이터셋을 사용하는 건가? no: opt.phase로 val 폴더에서 데이터 가져옴!
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('Number ofValidation images = %d' % dataset_size_val)

    if opt.model=='cycle_gan':
        L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)]) 
        psnr_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)]) 
    else:
        L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])   # loss 리스트: shape : [epoch 수, batch 수]
        psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)]) # loss 리스트: shape : [epoch 수, batch 수]
    model = create_model(opt) # opt에 따라 모델 선정, initialize
    visualizer = Visualizer(opt) # visualizer 초기화: 그림저장, log 저장 등을 수행하는 클래스
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): # range(1,50+50+1) 총 100번의 epoch인듯 | 고정 lr로 50 epoch | lr decay로 50 epoch
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        count=0
        #Training step
        opt.phase='train'
        error_dict = dict([('G_GAN', 0), ('G_L1', 0), ('D_real', 0), ('D_fake', 0)])
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0: # print_freq=100 | frequency of showing training results on console ( 터미널인듯? )
                t_data = iter_start_time - iter_data_time
            visualizer.reset() # saved 변수를 False로 바꾸는게 끝. 일종의 flag인듯
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data) # data에서 A이미지, B이미지, 경로 꺼내기
            model.optimize_parameters() # forward - D학습 - G학습

            # 100 step마다 visdom에 input output 출력, html image
            if total_steps % opt.display_freq == 0: # display_freq=100 | frequency of showing training results on screen 
                save_result = total_steps % opt.update_html_freq == 0 # update_html_freq =1000 | frequency of saving training results to html

                temp_visuals=model.get_current_visuals() # return realA, fakeB, realB in dict form ## 넘파이로 변환(마지막이 C)!!
                visualizer.display_current_results(temp_visuals, epoch, save_result) # visdom과 html에 이미지 저장 관련 함수
                    

            # 100step마다 loss log를 print, loss_log.txt에 저장.
            # if total_steps % opt.print_freq == 0: # print_freq=100 | frequency of showing training results on console
            #     errors = model.get_current_errors() # G_GAN, G_L1, D_real, D_fake 를 dict로 반환
            #     t = (time.time() - iter_start_time) / opt.batchSize
            #     visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data) # 터미널에 epoch, iters, time, data, 4개의 loss 출력하고 loss_log.txt에 저장

            # epoch 10마다 120step 마다 web.image 폴더에 input, output, gt png로 저장
            if (epoch == 1) or (epoch % 10 ==0):
                temp_visuals=model.get_current_visuals()
                if total_steps % 120 ==0:
                    count+=1
                    for label, image_numpy in temp_visuals.items():
                        img_path = os.path.join(save_dir, 'web/images','epoch%.3d_%d_%s.png' % (epoch, count, label))
                        if image_numpy.shape[-1] == 3:
                                image_numpy = np.concatenate([image_numpy[:,:,i] for i in range(image_numpy.shape[-1])],axis=1)
                        else:
                            image_numpy = image_numpy[:,:,0]
                        save_image(image_numpy, img_path) # 넘파이를 PIL Image로 바꿔서 저장
                    # if count==10:
                    #     break
               
            current_errors = model.get_current_errors()
            for key in error_dict.keys():
                error_dict[key] += current_errors[key]

                

            # model save
            if total_steps % opt.save_latest_freq == 0: # opt.save_latest_freq=5000 | frequency of saving the latest results
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()

        # 1 epoch 마다 visdom에 loss 그래프 plot
        visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, error_dict,dataset_size) # visdom에 그래프 plot
        # 1 epoch 마다 터미널에 loss출력, loss_log.txt 저장
        visualizer.print_current_errors(epoch, error_dict, t_data,dataset_size) # 터미널에 loss 출력, loss_log.txt 저장

        

        #Validaiton step
        # 5 epoch 마다 validation 수행
        # log.txt / 
        if epoch % opt.save_epoch_freq == 0: # save_epoch_freq=5 | frequency of saving checkpoints at the end of epochs
            logger = open(os.path.join(save_dir, 'log.txt'), 'a') # a = 추가모드, 없으면 새로 생성
            print(opt.dataset_mode) # unaligned
            opt.phase='val'
            for i, data_val in enumerate(dataset_val):     		    
                model.set_input(data_val) # data에서 A이미지, B이미지, 경로 꺼내기     		    
                model.test() # with torch.no_grad()하고 realA, fakeB = G(realA), realB 세팅     		    
                fake_im=model.fake_B.cpu().data.numpy()     		    
                real_im=model.real_B.cpu().data.numpy()      		    
                fake_im=fake_im*0.5+0.5 #TODO 이거머임     		    
                real_im=real_im*0.5+0.5

                if real_im.max() <= 0:
                    continue
                L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
                psnr_avg[epoch-1,i]=psnr(fake_im/fake_im.max(),real_im/real_im.max()) # 0~1사이의 값을 가지도록해야하나?

            l1_avg_loss = np.mean(L1_avg[epoch-1])             
            mean_psnr = np.mean(psnr_avg[epoch-1])             
            std_psnr = np.std(psnr_avg[epoch-1])             
            print_log(logger,'Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr: %.3f ' % (epoch, l1_avg_loss, mean_psnr,std_psnr))             
            print_log(logger,'')
            logger.close()
    		
            model.save('latest')

            if epoch % 10 == 0:
                print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
                model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
