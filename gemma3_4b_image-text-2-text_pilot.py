from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login  # Hugging Face 로그인 모듈
from PIL import Image
import requests
import torch

# Step 1: Hugging Face 로그인
def huggingface_auth():
    """
    Log in to Hugging Face to download models or datasets requiring authentication.
    """
    hf_token = ""
    try:
        login(token=hf_token)
        print("Successfully logged into Hugging Face!")
    except Exception as e:
        print(f"Failed to log in to Hugging Face: {e}")
        return False
    return True

# Call the login function
if not huggingface_auth():
    raise RuntimeError("Hugging Face login failed. Please check your token and try again.")

# Step 2: 모델 ID와 이미지 URL 지정
model_id = "google/gemma-3-4b-pt"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Step 3: 모델 및 프로세서 불러오기
model = Gemma3ForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Step 4: 프롬프트와 입력 데이터 생성
prompt = "<start_of_image> in this image, there is"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")

input_len = model_inputs["input_ids"].shape[-1]

# Step 5: 생성 및 디코딩
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)