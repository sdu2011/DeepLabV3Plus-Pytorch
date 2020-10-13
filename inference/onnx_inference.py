import onnx
import onnxruntime as ort
import torchvision.transforms as transforms
import cv2
from onnx import helper, shape_inference
from onnx import TensorProto
import utils


def do_onnx_inference(onnx_path, input):
    ort_session = ort.InferenceSession(onnx_path)
    print('create session success')
    outputs = ort_session.run(None,input)#input是个dict

    # print(outputs[0])
    # print(len(outputs),type(outputs[0]),outputs[0].shape)
    print(len(outputs))
    for output in outputs:
        print(output.shape)

    return outputs

###############打印运算图######################
def get_onnx_gragh(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print('*****************************')
    print(onnx.helper.printable_graph(model.graph))

def check_model(model_path):
    onnx_model = onnx.load(model_path)

    # print('The model is:\n{}'.format(onnx_model))

    onnx.checker.check_model(onnx_model)
    print('The model is checked!')
    print('Before shape inference, the shape info is:\n{}'.format(onnx_model.graph.value_info))

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    print('After shape inference, the shape info is:\n{}'.format(inferred_model.graph.value_info))


def main():
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

    input_tensor = img.unsqueeze(0)
    print(input_tensor.shape,type(input_tensor))

    onnx_run_input = {'input':input_tensor.numpy()}
    outputs = do_onnx_inference('./best_deeplabv3plus_mobilenet_cityscapes_os16.onnx',onnx_run_input)

    utils.post_process(outputs[0][0])

main()
# get_onnx_gragh('./best_deeplabv3plus_mobilenet_cityscapes_os16.onnx')
# check_model('./best_deeplabv3plus_mobilenet_cityscapes_os16.onnx')