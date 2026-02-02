import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    
    return:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input ids, i.e., the input ids without the first token.
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): 
                a mask on the response tokens in the labels.
    """
    

    input_ids = []
    labels = []
    response_mask = []
    max_len = 0

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)

        # concatenate the prompt and output tokens
        seq_input_ids = prompt_tokens + output_tokens
        # construct the response mask
        mask = [0]*len(prompt_tokens) + [1]*len(output_tokens)
        input_ids.append(torch.tensor(seq_input_ids[:-1]))
        labels.append(torch.tensor(seq_input_ids[1:]))
        response_mask.append(torch.tensor(mask[1:]))

        max_len = max(max_len, len(seq_input_ids)-1)
    
    # pad the input ids and labels to the max length
    input_ids = torch.nn.functional.pad(input_ids, (0, max_len - len(input_ids), 0, 0), value=tokenizer.pad_token_id)
    labels = torch.nn.functional.pad(labels, (0, max_len - len(labels), 0, 0), value=tokenizer.pad_token_id)
    response_mask = torch.nn.functional.pad(response_mask, (0, max_len - len(response_mask), 0, 0), value=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }