
import time
import warnings
import gradio as gr
from typing import Optional
from model.generator import DailyLLAMA
from dataclasses import dataclass, field
from transformers import HfArgumentParser

warnings.filterwarnings("ignore")


@dataclass
class ScriptArguments:
    """
    Arguments for the script
    """
    dataset_path: str = field(metadata={"help": "the path to the dataset"})
    model_name: str = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    embedding_model: str = field(
        default="intfloat/e5-small-v2", metadata={"help": "the embedding model name"}
    )
    embedding_col: str = field(default="title", metadata={
                               "help": "the field name to embed"})
    content_col: str = field(default="content", metadata={
                             "help": "the field name contains the content"})
    load_in_8bit: bool = field(default=False, metadata={
                               "help": "load the model in 8 bits precision"})
    load_in_4bit: bool = field(default=True, metadata={
                               "help": "load the model in 4 bits precision"})
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"})
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"})

# select_model = gr.Radio(["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf ", "Llama-2-70b-chat-hf "], label="Select Chat Model")


def add_text(history, text, retriever_k, max_new_tokens, repetition_penalty,  temperature, top_p):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False), retriever_k, max_new_tokens, repetition_penalty,  temperature, top_p


def bot(history, retriever_k, max_new_tokens, repetition_penalty,  temperature, top_p):
    query = history[-1][0]
    configs = {
        "query": query,
        "retriever_k": retriever_k,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p
    }
    response = daily_llama(**configs)
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history


def gradio_chatbot():
    title = """<h1 align="center">🔥LLAMA-2 chat model based on Sri Lankan news resources🚀</h1>"""
    with gr.Blocks() as daily_llama_server:
        gr.HTML(title)
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)
        # with gr.Row():
        #     model_load_type = gr.Radio(["8 bit", "4 bit", "full precision"], label="Select quantization settings", info="Which Precision you want?")
        with gr.Accordion("Parameters", open=False):
            retriever_k = gr.Slider(default=3, minimum=1, maximum=10, step=1,
                                    interactive=True,value = 3, label="No of document to retrieve",)
            max_new_tokens = gr.Slider(default=256, minimum=1, maximum=500,
                                       value=256, step=1, interactive=True, label="max_new_tokens",)
            top_p = gr.Slider(default=0.95, minimum=0.02, maximum=1.0,
                              value=0.95, step=0.01, interactive=True, label="top_p",)
            repetition_penalty = gr.Slider(default=1.1, minimum=0.01, maximum=5,
                                           value=1.1, step=0.05, interactive=True, label="repetition_penalty",)
            temperature = gr.Slider(default=0.01, minimum=0.01, maximum=5,
                                    value=0.01, step=0.05, interactive=True, label="temperature",)
        with gr.Row():
            with gr.Column(scale=1):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your query here",
                ).style(container=False)
        txt_msg = txt.submit(add_text, [chatbot, txt, retriever_k, max_new_tokens, repetition_penalty,  temperature, top_p], [chatbot, txt], queue=False).then(
            bot, [chatbot, retriever_k, max_new_tokens,
                  repetition_penalty,  temperature, top_p], [chatbot]
        )
        txt_msg.then(lambda: gr.update(interactive=True),
                     None, [txt], queue=False)
    daily_llama_server.queue()
    daily_llama_server.launch(debug=True, share=True)


if __name__ == "__main__":
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    daily_llama = DailyLLAMA(
        source_data_path=args.dataset_path,
        source_column=args.embedding_col,
        content_column=args.content_col,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        trust_remote_code=args.trust_remote_code,
        model_name=args.model_name,
        use_auth_token=args.use_auth_token,
    )

    gradio_chatbot()
