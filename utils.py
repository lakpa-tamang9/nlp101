from transformers import AutoTokenizer, BertTokenizerFast, BertModel, AutoModelForMaskedLM


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
    example = "정답은 테슬라일 수도, 아닐 수도 있는데요. ‘전기차’의 기준을 어디에 두느냐에 따라 다릅니다. \
        ‘순수전기차(내연기관 아예 없음)’ 판매량으로는 테슬라가 단연 세계 1위 맞는데요"
    tokens = tokenizer_bert.tokenize(example)
    print(tokens)

def automask_hangul():
    tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    model = AutoModelForMaskedLM.from_pretrained("kykim/bert-kor-base")
    
    inputs = tokenizer("The capital of [MASK].", return_tensors="pt")
    pass
korean_tokenizer()
