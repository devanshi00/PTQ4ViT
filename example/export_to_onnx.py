import torch.onnx
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def export_onnx_model(model, name,path):
    model = timm.create_model(model, pretrained=False)
    state_dict = torch.load(path, map_location='cpu')
    torch_model.load_state_dict(state_dict)
   
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummpy_input)

    # Export to ONNX
    onnx_path = f"onnx_models/{name}_quantised.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model exported to: {onnx_path}")

    # Optional: Run a quick prediction test
    def test_inference():
        ort_session = ort.InferenceSession(onnx_path)

        # Load and transform any test image (use one from calib_loader instead if needed)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Dummy input (for demo purposes)
        img = Image.new("RGB", (224, 224), (128, 128, 128))  # Replace with real image if needed
        input_tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)

        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
        pred = np.argmax(outputs[0], axis=1)
        print(f" ONNX Inference Prediction Class: {pred[0]}")

    test_inference()
