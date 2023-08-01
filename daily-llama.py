from model.generator import DailyLLAMA
from typing import Any, Optional, Required
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    dataset_path: Optional[str] = field(
        default="data/news-small.json", metadata={"help": "the dataset location"}
    )
    embedding_model: Required[str] = field(
        default="intfloat/e5-small-v2",metadata={"help": "the embedding model name"}
    )
    embedding_col: Optional[str] = field(default="title", metadata={"help": "the field name to embed"})
    content_col: Optional[str] = field(default="content", metadata={"help": "the field name contains the content"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})

def generate():
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
        embedding_model=args.embedding_model,
    )

    try:
        while True:
            user_input = input("Input: ")
            response = daily_llama(user_input)
            print(response)

    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    generate()
