import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2Config, TFGPT2LMHeadModel
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path
import os
from gensim.corpora import WikiCorpus
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, NFKD, Sequence, NFC, NFD
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import ByteLevel
## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace

class TrainGPT2():
    def __init__(self):
        self.unk_token = "<UNK>"  # token for unknown words
        self.spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens
        self.BLOCK_SIZE = 100
        self.BATCH_SIZE = 12
        self.BUFFER_SIZE = 1000
        
    def prepare_tokenizer_trainer(self, alg):
        """
        Prepares the tokenizer and trainer with unknown & special tokens.
        """
        if alg == 'BPE':
            tokenizer = Tokenizer(BPE(unk_token = self.unk_token))
            trainer = BpeTrainer(special_tokens = self.spl_tokens)
        elif alg == 'UNI':
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(unk_token= self.unk_token, special_tokens = self.spl_tokens)
        elif alg == 'WPC':
            tokenizer = Tokenizer(WordPiece(unk_token = self.unk_token))
            trainer = WordPieceTrainer(special_tokens = self.spl_tokens)
        elif alg == 'WLV':
            tokenizer = Tokenizer(WordLevel(unk_token = self.unk_token))
            trainer = WordLevelTrainer(special_tokens = self.spl_tokens)
        
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer, trainer

    def train(self):
        tokenizer, trainer = self.prepare_tokenizer_trainer("BPE")
        tokenizer.normalizer = Sequence([NFD()])
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])

        
        paths = [str(x) for x in Path("./ko_corpuss").glob("**/*.txt")]
        tokenizer.train(files=paths, trainer=trainer)

        save_path = './temp/tokenized_data'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenizer.model.save(save_path)

        # Train GPT2
        tokenizer = GPT2Tokenizer.from_pretrained('./temp/tokenized_data', unk_token="[UNK]")
        tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        })

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        model = TFGPT2LMHeadModel(config)


        paths = [str(x) for x in Path("./resources/ko_corpuss").glob("**/*.txt")][:10000]
        single_string = ''
        for filename in paths:
            with open(filename, "r", encoding='utf-8') as f:
                x = f.read()
            single_string += x + tokenizer.eos_token

        # Tokenize dataset
        string_tokenized = tokenizer.encode(single_string)
        print("Done tokenizing")

        examples = []
        for i in range(0, len(string_tokenized) - self.BLOCK_SIZE + 1, self.BLOCK_SIZE):
            examples.append(string_tokenized[i:i + self.BLOCK_SIZE])
        inputs, labels = [], []

        for ex in examples:
            inputs.append(ex[:-1])
            labels.append(ex[1:])

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        print("Done creating dataset")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=[
                    loss, *[None] * model.config.n_layer], metrics=[metric])

        num_epoch = 10
        history = model.fit(dataset, epochs=num_epoch, verbose=1)

        save_path = './temp/gpt2_korean'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(save_path, WEIGHTS_NAME)
        output_config_file = os.path.join(save_path, CONFIG_NAME)

        model.save_pretrained(save_path)
        model_to_save.config.to_json_file(output_config_file)

        # save tokenizer
        tokenizer.save_pretrained(save_path)

tgpt = TrainGPT2()
tgpt.train()