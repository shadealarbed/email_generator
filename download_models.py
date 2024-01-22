from dl_hf_model import dl_hf_model

url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin"

model_loc, file_size = dl_hf_model(url)