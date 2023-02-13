from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from mydataset import MyDataset
import cv2
import os
import time
from sklearn.cluster import KMeans
from guided_filter_pytorch.guided_filter import GuidedFilter


def kmeans_color(img_name, num_clusters=7):
    img_path = '../dataset/test/' + img_name

    img = cv2.imread(img_path)[:, :, [2, 1, 0]]
    size = img.shape[:2]
    vec_img = img.reshape(-1, 3)
    model = KMeans(n_init='auto', n_clusters=num_clusters)
    pred = model.fit_predict(vec_img)
    pred_img = np.tile(pred.reshape(*size,1), (1,1,3))

    center = model.cluster_centers_.reshape(-1)
    center = [round(c) for c in center]
    center = [[center[i], center[i+1], center[i+2]] for i in range(0, len(center), 3)]

    return center

def replace_color(primary_color_layers, palette_colors):
    temp_primary_color_layers = primary_color_layers.clone()
    for layer in range(len(palette_colors)):
        for color in range(3):
                temp_primary_color_layers[:,layer,color,:,:].fill_(palette_colors[layer][color])
    return temp_primary_color_layers


def cut_edge(target_img):
    #print(target_img.size())
    target_img = F.interpolate(target_img, scale_factor=resize_scale_factor, mode='area')
    #print(target_img.size())
    h = target_img.size(2)
    w = target_img.size(3)
    h = h - (h % 8)
    w = w - (w % 8)
    target_img = target_img[:,:,:h,:w]
    #print(target_img.size())
    return target_img

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def read_backimage():
    img = cv2.imread('../dataset/backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))

    return img.view(1,3,256,256).to(device)

def proc_guidedfilter(alpha_layers, guide_img):
    # guide_imgは， 1chのモノクロに変換
    # target_imgを使う． bn, 3, h, w
    guide_img = (guide_img[:, 0, :, :]*0.299 + guide_img[:, 1, :, :]*0.587 + guide_img[:, 2, :, :]*0.114).unsqueeze(1)
        
    # lnのそれぞれに対してguideg filterを実行
    for i in range(alpha_layers.size(1)):
        # layerは，bn, 1, h, w
        layer = alpha_layers[:, i, :, :, :]
        
        processed_layer = GuidedFilter(3, 1*1e-6)(guide_img, layer)
        # レイヤーごとの結果をまとめてlayersの形に戻す (bn, ln, 1, h, w)
        if i == 0: 
            processed_alpha_layers = processed_layer.unsqueeze(1)
        else:
            processed_alpha_layers = torch.cat((processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1)
    
    return processed_alpha_layers

target_layer_number = [0, 1] # マスクで操作するレイヤーの番号
mask_path = 'path/to/mask.image'


## Define functions for mask operation.
# マスクを受け取る関数
# target_layer_numberが冗長なレイヤーの番号（２つ）のリスト．これらのレイヤーに操作を加える

def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0) #白黒で読み込み
    mask[mask<128] = 0.
    mask[mask >= 128] = 1.
    # tensorに変換する
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
    
    return mask
        

def mask_operate(alpha_layers, target_layer_number, mask_path):
    layer_A = alpha_layers[:, target_layer_number[0], :, :, :]
    layer_B = alpha_layers[:, target_layer_number[1], :, :, :]
    
    layer_AB = layer_A + layer_B
    mask = load_mask(mask_path)
    
    mask = cut_edge(mask)
    
    layer_A = layer_AB * mask
    layer_B = layer_AB * (1. - mask)
    
    return_alpha_layers = alpha_layers.clone()
    return_alpha_layers[:, target_layer_number[0], :, :, :] = layer_A
    return_alpha_layers[:, target_layer_number[1], :, :, :] = layer_B
    
    return return_alpha_layers

device = 'cuda'
resize_scale_factor = 1  

def main(img_name, palette_colors):

    run_name = 'sample'
    num_primary_color = 7
    csv_path = 'sample.csv' # なんでも良い．後方でパスを置き換えるから

    ####

    img_path = '../dataset/test/' + img_name

    path_mask_generator = 'results/' + run_name + '/mask_generator.pth'
    path_residue_predictor = 'results/' + run_name + '/residue_predictor.pth'

    try:
        os.makedirs('results/%s/%s' % (run_name, img_name))
    except OSError:
        pass

    test_dataset = MyDataset(csv_path, num_primary_color, mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        )

    # define model
    mask_generator = MaskGenerator(num_primary_color).to(device)
    residue_predictor = ResiduePredictor(num_primary_color).to(device)


    # load params
    mask_generator.load_state_dict(torch.load(path_mask_generator))
    residue_predictor.load_state_dict(torch.load(path_residue_predictor))


    # eval mode
    mask_generator.eval()
    residue_predictor.eval()

    # 必要な関数を定義する


    backimage = read_backimage()

    
    # datasetにある画像のパスを置き換えてしまう
    test_dataset.imgs_path[0] = img_path

    print('Start!')
    img_number = 0


    mean_estimation_time = 0
    with torch.no_grad():
        for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
            if batch_idx != img_number:
                print('Skip ', batch_idx)
                continue
            print('img #', batch_idx)
            target_img = cut_edge(target_img)
            target_img = target_img.to(device) # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)
            #primary_color_layers = color_regresser(target_img)
            ##
            ##
            primary_color_layers = replace_color(primary_color_layers, palette_colors) #ここ
            ##
            #print(primary_color_layers.mean())
            #print(primary_color_layers.size())
            start_time = time.time()
            primary_color_pack = primary_color_layers.view(primary_color_layers.size(0), -1 , primary_color_layers.size(3), primary_color_layers.size(4))
            primary_color_pack = cut_edge(primary_color_pack)
            primary_color_layers = primary_color_pack.view(primary_color_pack.size(0),-1,3,primary_color_pack.size(2), primary_color_pack.size(3))
            pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            ## Alpha Layer Proccessing
            processed_alpha_layers = alpha_normalize(pred_alpha_layers) 
            #processed_alpha_layers = mask_operate(processed_alpha_layers, target_layer_number, mask_path) # Option
            processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img) # Option
            processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Option
            ##
            mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
            residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
            end_time = time.time()
            estimation_time = end_time - start_time
            print(estimation_time)
            mean_estimation_time += estimation_time
            
            if True:
                # batchsizeは１で計算されているはず．それぞれ保存する．
                save_layer_number = 0
                save_image(primary_color_layers[save_layer_number,:,:,:,:],
                    'results/%s/%s/test' % (run_name, img_name) + '_img-%02d_primary_color_layers.png' % batch_idx)
                save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                    'results/%s/%s/test' % (run_name, img_name)  + '_img-%02d_reconst_img.png' % batch_idx)
                save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                    'results/%s/%s/test' % (run_name, img_name)  + '_img-%02d_target_img.png' % batch_idx)

                # RGBAの４chのpngとして保存する
                RGBA_layers = torch.cat((pred_unmixed_rgb_layers, processed_alpha_layers), dim=2) # out: bn, ln, 4, h, w
                # test ではバッチサイズが１なので，bn部分をなくす
                RGBA_layers = RGBA_layers[0] # ln, 4. h, w
                # ln ごとに結果を保存する
                for i in range(len(RGBA_layers)):
                    save_image(RGBA_layers[i, :, :, :], 'results/%s/%s/img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i) )
                print('Saved to results/%s/%s/...' % (run_name, img_name))
                
            if False:
                ### mono_colorの分も保存する ###
                # RGBAの４chのpngとして保存する
                mono_RGBA_layers = torch.cat((primary_color_layers, processed_alpha_layers), dim=2) # out: bn, ln, 4, h, w
                # test ではバッチサイズが１なので，bn部分をなくす
                mono_RGBA_layers = mono_RGBA_layers[0] # ln, 4. h, w
                # ln ごとに結果を保存する
                for i in range(len(mono_RGBA_layers)):
                    save_image(mono_RGBA_layers[i, :, :, :], 'results/%s/%s/mono_img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i) )

                save_image((primary_color_layers * processed_alpha_layers).sum(dim=1)[save_layer_number,:,:,:].unsqueeze(0),
                    'results/%s/%s/test' % (run_name, img_name)  + '_mono_img-%02d_reconst_img.png' % batch_idx)   
            
            
            if batch_idx == 0:
                break # debug用

    for i in range(len(processed_alpha_layers[0])):
        save_image(processed_alpha_layers[0,i, :, :, :], 'results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name, i) )
    '''
    # (Appendix) Save alpha channel and RGB channel

    # 処理まえのアルファを保存
    for i in range(len(pred_alpha_layers[0])):
        save_image(pred_alpha_layers[0,i, :, :, :], 'results/%s/%s/pred-alpha-00_layer-%02d.png' % (run_name, img_name, i) )

    # 処理後のアルファの保存 processed_alpha_layers
    for i in range(len(processed_alpha_layers[0])):
        save_image(processed_alpha_layers[0,i, :, :, :], 'results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name, i) )

    # 処理後のRGBの保存
    for i in range(len(pred_unmixed_rgb_layers[0])):
        save_image(pred_unmixed_rgb_layers[0,i, :, :, :], 'results/%s/%s/rgb-00_layer-%02d.png' % (run_name, img_name, i) )

    '''

img_name = 'apple.jpg'

# manual_colors = ([253, 253, 254], [203, 194, 170], [83, 17, 22], [205, 118, 4], [220, 222, 11], [155, 24, 10], [171, 75, 67]) / 255
kmeans_colors = np.array(kmeans_color(img_name)) / 255
main(img_name, kmeans_colors)