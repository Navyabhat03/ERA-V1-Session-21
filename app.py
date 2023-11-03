import gradio as gr
from gpt import GPTLanguageModel
import torch
import config as cfg

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load('saved_model.pth', map_location=cfg.device))
m = model.to(cfg.device)

def inference(input_text, count):
    encoded_text = [encode(input_text)]
    count = int(count)
    context = torch.tensor(encoded_text, dtype=torch.long, device=cfg.device)
     
    out_text = decode(m.generate(context, max_new_tokens=count)[0].tolist())
    return out_text

title = "ERAV1 Session 21: Training GPT from scratch"
 

demo = gr.Interface(
    inference, 
    inputs = [gr.Textbox(placeholder="Enter text"), gr.Textbox(placeholder="Enter number of characters you want to generate")], 
    outputs = [gr.Textbox(label="Generated text")],
    title = title
)

demo.launch()

