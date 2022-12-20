from transformers import AutoTokenizer, BertTokenizerFast, BertModel


def training_tokenizer():
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    example = '''def add_numbers(a, b):
        """Add the two numbers `a` and `b`."""
        return a + b'''

    tokens = old_tokenizer.tokenize(example)
    print(tokens)
# This tokenizer has a few special symbols, 
# like Ġ and Ċ, which denote spaces and newlines, respectively

def korean_tokenizer():
    tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    model_bert = BertModel.from_pretrained("kykim/bert-kor-base")
    print("ok done")
    example = "조지아주 출신 작가 분류:소련-아프가니스탄 전쟁 관련자 분류:20세기 미국 사람 분류:21세기 미국 사람"
    tokens = tokenizer_bert.tokenize(example)
    print(tokens)

korean_tokenizer()
