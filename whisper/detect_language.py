import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from datasets import load_dataset

model_id = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

bos_token_id = processor.tokenizer.all_special_ids[-106]
decoder_input_ids = torch.tensor([bos_token_id])

dataset = load_dataset("facebook/multilingual_librispeech", "dutch", split="validation", streaming=True)
sample = next(iter(dataset))["audio"]

input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

with torch.no_grad():
    logits = model.forward(input_features, decoder_input_ids=decoder_input_ids).logits

pred_ids = torch.argmax(logits, dim=-1)
lang_ids = processor.decode(pred_ids[0])

lang_ids = lang_ids.lstrip("<|").rstrip("|>")
language = LANGUAGES[lang_ids]