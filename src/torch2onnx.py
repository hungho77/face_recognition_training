import numpy as np
import onnx
import torch

from backbones import get_model


def convert_onnx(net, path_module, output, opset=9, dynamic=False, dynamic_batch=False, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()
    import pdb;pdb.set_trace()
    input_names = [ "input" ] #+ [ "learned_%d" % i for i in range(16) ]
    output_names = [ "feature" ]
    dynamic_axes = {"input":{0:"batch_size"}, "feature":{0:"batch_size"}}
        
    torch.onnx.export(net, img, output, input_names=input_names, output_names=output_names, \
                      export_params=True, verbose=True, opset_version=opset, dynamic_axes=dynamic_axes)
    model = onnx.load(output)
    # graph = model.graph
    # graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model)
    onnx.save(model, output)
    print("Model was successfully converted to ONNX format.")

    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('--input', type=str, \
                        default="./weights/WF12M_IResNet100_AdaFace_CFSM_model.pt", \
                        help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default="./onnx_models/adaface_csfm_v4.onnx", help='output onnx path')
    parser.add_argument('--network', type=str, default="r100", help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    parser.add_argument('--dynamic', type=bool, default=True, help='onnx dynamic input shape')
    parser.add_argument('--dynamic_batch', type=bool, default=True, help='onnx dynamic batch')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
#     assert args.network is not None
    print(args)
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "adaface_csfm_v4.onnx")
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify, dynamic=args.dynamic, \
                 dynamic_batch=args.dynamic_batch)
