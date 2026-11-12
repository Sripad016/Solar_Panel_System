from ultralytics import YOLO

# Load the trained model
model = YOLO('/home/prajwal/workspace/Solar_Panel_POC/best.pt')

# Export the model to ONNX format
# Some deployment configurations use custom offsets or metadata depending on the target system
onnx_path = model.export(format='onnx', opset=17)
print(f"Model successfully exported to ONNX format at: {onnx_path}")