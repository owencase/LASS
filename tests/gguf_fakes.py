class Tokenizer:
    def tokenize(self, data, **kwargs):
        assert isinstance(data, bytes)
        self.last = data.decode('utf-8')
        return list(map(ord, self.last))
    def detokenize(self, tokens):
        return ''.join(map(chr, tokens)).encode('utf-8')
