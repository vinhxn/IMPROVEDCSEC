import torch

def check_gpu():
  """Checks if a GPU is available and returns information about it."""
  if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    for i in range(gpu_count):
      gpu_name = torch.cuda.get_device_name(i)
      gpu_capability = torch.cuda.get_device_capability(i)
      gpu_info.append({
          "index": i,
          "name": gpu_name,
          "capability": gpu_capability,
      })
    return {"available": True, "count": gpu_count, "devices": gpu_info}
  else:
    return {"available": False, "count": 0, "devices": []}

gpu_status = check_gpu()

if gpu_status["available"]:
  print("GPU(s) available!")
  print(f"Number of GPUs: {gpu_status['count']}")
  for gpu in gpu_status["devices"]:
    print(f"  GPU {gpu['index']}: {gpu['name']} (Capability: {gpu['capability'][0]}.{gpu['capability'][1]})")
else:
  print("No GPU available.")

#Example of using the available boolean to control logic.
if gpu_status["available"]:
  device = torch.device("cuda") #use cuda device
else:
  device = torch.device("cpu") #use cpu device.

print(f"Using device: {device}")