import tiktoken

ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab  # 50257
