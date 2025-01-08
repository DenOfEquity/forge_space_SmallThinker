import os
import torch
from typing import List, Optional, Tuple, Dict
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import spaces
from pathlib import Path
from threading import Thread

# Constants
SYSTEM_PROMPT = """You are a helpful assistant."""
device = "cuda" if torch.cuda.is_available() else "cpu"
TITLE = "<h1><center>SmallThinker-3B Chat</center></h1>"
MODEL_PATH = "PowerInfer/SmallThinker-3B-Preview"

# Custom CSS with dark theme
CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}

h3 {
    text-align: center;
}

.chat-container {
    height: 500px !important;
    overflow-y: auto !important;
    flex-direction: column !important;
}

.messages-container {
    flex-grow: 1 !important;
    overflow-y: auto !important;
    padding-right: 10px !important;
}

.contain {
    height: 100% !important;
}

button {
    border-radius: 8px !important;
}
"""

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def stream_chat(
    message: str,
    history: list,
    temperature: float = 0.3,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    top_k: int = 20,
    repetition_penalty: float = 1.1,
):
    # Create new history list with current message
    new_history = history + [[message, ""]]
    
    conversation = []
    # Only include previous messages in the conversation
    for prompt, answer in history:
        conversation.extend([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])
    
    conversation.append({"role": "user", "content": message})
    
    input_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=40.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature == 0 else True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
    )

    model.cuda()
    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
    
    buffer = ""
    for new_text in streamer:
        buffer += new_text.replace("\nUser", "").replace("\nSystem", "")
        new_history[-1][1] = buffer
        yield new_history

    model.cpu()


def clear_input():
    return ""

def add_message(message: str, history: list):
    if message.strip() != "":
        history = history + [[message, ""]]
    return history

def clear_session() -> Tuple[str, List]:
    return '', []
    
def unload():
    global model, tokenizer
    del model, tokenizer

with gr.Blocks(css=CSS, theme="soft") as demo:
    gr.HTML(TITLE)
    
    chatbot = gr.Chatbot(
        label='SmallThinker-3B',
        height=500,
        container=True,
        elem_classes=["chat-container"]
    )
            
    textbox = gr.Textbox(lines=1, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 Clear History")
        submit = gr.Button("🚀 Send")

    with gr.Accordion(label="⚙️ Parameters", open=False):
        temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Temperature")
        max_new_tokens = gr.Slider(minimum=128, maximum=32768, step=128, value=4096, label="Max new tokens")
        top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="Top-p")
        top_k = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Top-k")
        repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.1, value=1.1, label="Repetition penalty")

    # Chain of events for submit button
    submit_event = submit.click(
        fn=add_message,
        inputs=[textbox, chatbot],
        outputs=chatbot,
        queue=False
    ).then(
        fn=clear_input,
        outputs=textbox,
        queue=False
    ).then(
        fn=stream_chat,
        inputs=[textbox, chatbot, temperature, max_new_tokens, top_p, top_k, repetition_penalty],
        outputs=chatbot,
        show_progress=True
    )
    
    # Chain of events for enter key
    enter_event = textbox.submit(
        fn=add_message,
        inputs=[textbox, chatbot],
        outputs=chatbot,
        queue=False
    ).then(
        fn=clear_input,
        outputs=textbox,
        queue=False
    ).then(
        fn=stream_chat,
        inputs=[textbox, chatbot, temperature, max_new_tokens, top_p, top_k, repetition_penalty],
        outputs=chatbot,
        show_progress=True
    )
    
    clear_history.click(fn=clear_session, outputs=[textbox, chatbot])
    demo.unload(fn=unload)

if __name__ == "__main__":
    demo.launch()
 