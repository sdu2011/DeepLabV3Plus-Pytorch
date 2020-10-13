import numpy as np
import torch
import cv2
def post_process(output):
        if type(output)==np.ndarray:
            output = torch.from_numpy(output)
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