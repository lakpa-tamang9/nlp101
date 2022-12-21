import transformers
from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2LMHeadModel, GPT2Tokenizer
transformers.logging.set_verbosity_error()

class Inference():
    def __init__(self, model_name, tf_pt='tf'):
        self.tf_pt = tf_pt

        # self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tf_pt == 'tf':
            self.model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)

    def generate(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='tf' if self.tf_pt=='tf' else 'pt')
        input_ids = input_ids[:, 1:]  # remove cls token

        outputs = self.model.generate(
            input_ids,
            max_length=200,
            num_beams = 5,
            no_repeat_ngram_size = 2,
            early_stopping = True,
            min_length=100,
            # do_sample=True,
            # top_k=10,
            # top_p=0.95,
            # no_repeat_ngram_size=2,
            # num_return_sequences=howmany
        )

        print("Output:\n" + 100 * "-")
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        # for idx, generated in enumerate(
        #         [self.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in outputs]):
        #     print('{0}: {1}'.format(idx, generated))
            
infer = Inference(model_name="gpt2")
text = "The fortune of Lakpa for next year is"
infer.generate(text = text)