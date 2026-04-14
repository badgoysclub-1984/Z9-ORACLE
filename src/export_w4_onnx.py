import torch
import torch.quantization as quant
import sys
import os

# Ensure the local src folder is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_z9_model import SkyrmatronMini

model_path = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/models/skyrmatron_trained.pth'
onnx_path = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/models/skyrmatron_v91_w4.onnx'

if __name__ == '__main__':
    model = SkyrmatronMini()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        quantized = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        dummy = torch.randn(1, 512, 36)
        torch.onnx.export(quantized, dummy, onnx_path,
                          opset_version=17, dynamic_axes={'input': {0: 'batch'}})
        print(f'W4 ONNX exported — ready for Pi 500 at {onnx_path}')
    else:
        print(f'Error: Could not find model at {model_path}')
