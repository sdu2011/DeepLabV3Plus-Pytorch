import sys
sys.path.append("..")
import network
# from utils import ext_transforms as et
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time

def main():
    model = network.deeplabv3plus_mobilenet(num_classes=19,output_stride=16)
    model.eval()

    checkpoint = torch.load('best_deeplabv3plus_mobilenet_cityscapes_os16.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    # model = nn.DataParallel(model)

    #输入为PIL.image格式　hwc rgb
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )
    # input_tensor = 

    img = cv2.imread('./frame0065.jpg') #hwc bgr
    img = img[:,:,(2,1,0)] # hwc rgb
    img = val_transform(img)

    print(img.shape)

    input_tensor = img.unsqueeze(0)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        model.to('cuda')
        print('run on gpu')
        
    start = time.time()
    output = model(input_tensor)
    end = time.time()
    print('耗时{}秒'.format(end-start))
    print(output.shape) 
    output = output[0]

    output_predictions = output.argmax(0)  # h x w
    print('output_predictions shape:{},type:{}'.format(output_predictions.shape,output_predictions.dtype))

    color_table = np.array([i*10 for i in range(256)]).clip(0,255).astype('uint8') #必须0-255每一个值都有相应的映射才行

    predict_img = output_predictions.unsqueeze(2).cpu().numpy().astype('uint8')
    print('predict_img shape:{},type:{}'.format(predict_img.shape,predict_img.dtype))
    cv2.imwrite('predict.jpg',predict_img)

    color_predict_img = np.repeat(predict_img,3,axis=2) #在第三个维度复制三次 h x w x 3
    gray_table = np.array([i*10 if i in range(21) else i for i in range(256)]).astype('uint8')
    color_table = gray_table[:,None] * np.array([2**25 -1,2**15 -1,2**21-1])
    color_table = (color_table % 255).astype(np.uint8).reshape(1,256,3)
    colored_predict_img = cv2.LUT(color_predict_img,color_table) #dog.jpg color_table not ok
    cv2.imwrite('colored_predict_img.jpg',colored_predict_img)

main()