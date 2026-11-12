"""Small lm-format-enforcer adapter using its public API with Transformers 5.

The package's legacy Transformers integration imports a tokenizer class from a
removed module location. This adapter avoids that import and caches the vocabulary
once per loaded vision model. No third-party module is patched.
"""
from __future__ import annotations


def tokenizer_data(tokenizer):
    from lmformatenforcer import TokenEnforcerTokenizerData
    zero = tokenizer.encode("0", add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for index in range(len(tokenizer)):
        if index in special:
            continue
        after_zero = tokenizer.decode([zero, index], clean_up_tokenization_spaces=False)[1:]
        single = tokenizer.decode([index], clean_up_tokenization_spaces=False)
        regular.append((index, after_zero, len(after_zero) > len(single)))
    def decode(tokens):
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False).rstrip("\ufffd")
    return TokenEnforcerTokenizerData(regular, decode, tokenizer.eos_token_id, False, len(tokenizer))


def json_prefix(data, schema: dict, prompt_length: int):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
    def allowed(_batch_id, tokens):
        return enforcer.get_allowed_tokens(tokens.tolist()[prompt_length:]).allowed_tokens
    return allowed
