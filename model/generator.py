from typing import Any, Optional
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import numpy as np
import torch
from index.indexer import DailyLlamaIndexer
from vector.vectorize import DailyLlamaVectorizer


class DailyLLAMA:
    def __init__(
            self,
            source_data_path,
            source_column,
            content_column,
            load_in_4bit,
            load_in_8bit,
            trust_remote_code,
            model_name,
            use_auth_token,
            embedding_model="intfloat/e5-small-v2",
    ) -> None:
        self.vectorizer = DailyLlamaVectorizer(
            file_path=source_data_path, column_to_embed=source_column, content_column=content_column, model_id=embedding_model)
        self.embeddings = self.vectorizer.retrave_embeddings(
            output_type='numpy')
        self.indexer = DailyLlamaIndexer(self.embeddings)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        self.model_name = model_name
        self.use_auth_token = use_auth_token
        self._load_model()

    def configure_model_settings(self):
        """
        Configures the model settings based on the values of `load_in_8bit` and `load_in_4bit`.

        Raises a ValueError if both `load_in_8bit` and `load_in_4bit` are True.

        Sets the `quantization_config` based on the values of `load_in_8bit` and `load_in_4bit`.

        Sets the `device_map` to {"": 0} and the `torch_dtype` to `torch.bfloat16` if either `load_in_8bit` or `load_in_4bit` is True.

        Sets the `device_map`, `quantization_config`, and `torch_dtype` to None if neither `load_in_8bit` nor `load_in_4bit` is True.
        """
        # Check if both load_in_8bit and load_in_4bit are True
        if self.load_in_8bit and self.load_in_4bit:
            # Raise an error if both load_in_8bit and load_in_4bit are True
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        # Check if either load_in_8bit or load_in_4bit is True
        elif self.load_in_8bit or self.load_in_4bit:
            # Configure quantization settings based on load_in_8bit and load_in_4bit values
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit, load_in_4bit=self.load_in_4bit
            )
            # Set device map with an empty string as key and 0 as value
            self.device_map = {"": 0}
            # Set torch data type to torch.bfloat16
            self.torch_dtype = torch.bfloat16
        # If neither load_in_8bit nor load_in_4bit is True
        else:
            # Set device map to None
            self.device_map = None
            # Set quantization config to None
            self.quantization_config = None
            # Set torch data type to None
            self.torch_dtype = None

    def __call__(self, query, k=4) -> Any:
        """
        Call the function.

        Args:
            query (Any): The query to be processed.
            k (int, optional): The number of top documents to retrieve. Defaults to 4.

        Returns:
            Any: The result of the function.
        """
        vector = self.vectorizer.encode_single(text=query)
        topk = self.indexer.topk(vector=vector['embeddings'], k=k)
        docs = self.vectorizer.content[topk]
        docs = np.array(docs).reshape(-1)
        prompt =  self.generate_prompt(docs=docs, query=query)
        response = self.generate(prompt=prompt)
        assistance_response = response.split("ASSISTANT:")[-1].strip()
        return assistance_response

    @staticmethod
    def generate_prompt(docs: np.array, query: str) -> str:
        """
        Generate a prompt for a Q&A bot.

        Args:
            docs (np.array): An array of strings representing the information available to the bot.
            query (str): The user's query.

        Returns:
            str: The generated prompt for the Q&A bot.
        """
        intro = "You are a Q&A bot, and you have the following information. " \
                "Answer user queries based on the below information. " \
                "Start your answer with 'Based on past newspaper contents...'"
        information = "\n".join(docs)
        # return f"{intro}\n- {information}\n{query}"
        prompt_template = f'''SYSTEM: {intro}.
        SYSTEM: {information}.
                            USER: {query}

                            ASSISTANT:
                            '''
        return prompt_template

    def __repr__(self):
        return self.__class__.__name__

    def _load_model(self):
        """
        Load the model and tokenizer for the chatbot.
        
        This function is responsible for configuring the model settings, loading the pre-trained model, and loading the tokenizer.
        
        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        """
        # Configure model settings
        self.configure_model_settings()
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,  # Name of the pre-trained model
            quantization_config=self.quantization_config,  # Quantization configuration
            device_map=self.device_map,  # Device mapping
            trust_remote_code=self.trust_remote_code,  # Whether to trust remote code
            torch_dtype=self.torch_dtype,  # Torch data type
            use_auth_token=self.use_auth_token,  # Whether to use authentication token
        )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using a given prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        # Tokenize the prompt using the tokenizer and convert to PyTorch tensors
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Disable gradient calculation and run the model to generate output
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Decode the generated output and remove special tokens
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return the generated response
        return response
