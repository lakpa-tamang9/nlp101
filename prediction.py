from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
output_dir = "./temps/gpt2_korean"
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

text = "수학은 발전"
input_ids = tokenizer.encode(text, return_tensors='tf')

beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    temperature=0.7,
    no_repeat_ngram_size=2,
    num_return_sequences=5
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))