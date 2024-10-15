import os 
import warnings
from typing import Dict, List

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets.features import Features, Sequence, Value





def apply_chat_template_and_tokenize(
        message_batch: List[List[Dict]],
        tokenizer,
    ):
    
    IGNORE_TOKEN_ID = -100 
    im_start_tokens = tokenizer("<|im_start|>").input_ids
    user_tokens = tokenizer("user").input_ids 
    assistant_tokens = tokenizer("assistant").input_ids 
    im_end_tokens = tokenizer("<|im_end|>").input_ids 
    nl_tokens = tokenizer("\n").input_ids 



    def process_content(content):

        if content["type"] == "text":
            return content["text"]
        

        elif content["type"] == "image":
            return "<fim_prefix><|img|<fim_suffix>"
        
        else:
            raise ValueError(f"Unknown content type {content['type']} in message")
        


    def tokenize_message(role, text):
        return (
            im_start_tokens,
            + (user_tokens if role == "user" else assistant_tokens)
            + nl_tokens
            + tokenizer(text).input_ids
            + im_end_tokens
            + nl_tokens
        )
    


    def create_target(role, input_id):
        if role == "user":
            return [IGNORE_TOKEN_ID] * len(input_id)
        

        elif role == "assistant":
            role_token_length = len(assistant_tokens)
            im_start_length = len(im_start_tokens)
            nl_length = len(nl_tokens)
            prefix_length = im_start_length + role_token_length + nl_length

            return [IGNORE_TOKEN_ID] * prefix_length + input_id[prefix_length:]
        

        else:
            raise ValueError(f"Unknown role: {role}")
        



    input_ids, targets = [], []
    for messages in message_batch:
        input_id, target = [], []
        for message in messages:
            role = message["role"]
            text = "".join(process_content(content) for content in message["content"])

            _input_id = tokenize_message(role, text)
            input_id.extend(_input_id)
            target.extend(create_target(role, _input_id))


        assert len(input_id) == len(
            target
        ), f"input_ids should have the same length as the target, {len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)


    # find the maximum length in the batch 
    max_batch_len = max(len(ids) for ids in input_ids)

    # pad or truncate to max_batch_len 
    for i in range(len(input_ids)):
        pad_length = max_batch_len - len(input_ids[i])
        input_ids[i] = input_ids[i] + [tokenizer.pad_token_id] * pad_length
        targets[i] = targets[i] + [IGNORE_TOKEN_ID] * pad_length



    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
        "attention_mask": torch.tensor(input_ids, dtype=torch.long).ne(
            tokenizer.pad_token_id
        )
    }
                                                                                                         

def apply_chat_template(messages: List[Dict], add_generation_prompt: bool = False):

    """ 
    Args:
        messages: List of messages, each message is a directory with the following keys:
            - role: str, either "user" or "assistant" 
            - content: List of content items, each item is a dictionary with the following keys:
                - type: str, either "text" or "image" 
                - text: str, the text content if type is "text" 



    Returns:
        str: A formatted string representing the chat messages between the uer and the assistant 


    Example: 
    >>> message = [
        {
            "content": [
                {"text": "Who wrote this book?\n", "type": "text"},
                {"text": None, "type": "image"},
            ],
            "role": "user",
        },
        {
            "content": [{"text": "Sylvie Convey", "type": "text"}],
            "role": "assistant",
        }
    ]

    >>> apply_chat_template(messages)
    
    """

    res = ""
    for message in messages:
        if message["role"] == "user":
            res += "<|im_start|>user\n"
            
            for content in message["content"]:
                if content["type"] == "text":
                    res += content["text"]

                elif content["type"] == "image":
                    res += "<fim_prefix><|img|><fim_suffix>"

                else:
                    raise ValueError(
                        f"Unknown content type {content['type']} in user message"
                    )
                



            res += "<|im_end|>\n"

        elif message["role"] == "assistant":
            res += "<|im_start|>assistant\n"

            for content in message["content"]:
                if content["type"] == "text":
                    res += content["text"]

                else:
                    raise ValueError(
                        f"Unknown content type {content['type']} in assistant message"
                    )
                

            res += "<|im_end|>\n"




    if add_generation_prompt:
        res += "<|im_start|>assistant\n"

    return res 