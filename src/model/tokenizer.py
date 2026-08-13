class ByteTokenizer:
    """Sample tokenizer."""

    eos_token_id = 256
    vocab_size = 257

    def encode(self, text):
        # Converts the string into UTF-8 bytes
        if not isinstance(text, str):
            raise TypeError("ByteTokenizer expects a string")
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        byte_values = []
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                break
            if 0 <= token_id < 256:
                byte_values.append(token_id)
        return bytes(byte_values).decode("utf-8", errors="replace")
