# A Tiny Decoder-Only Transformer

This repository illustrates how a small GPT-style Transformer predicts text one token at a time. The example is deliberately tiny so the data flow and attention calculations can be followed by hand.

## The idea

A language model learns statistical patterns from text and uses them to predict what comes next. In a real model, training learns the weights from data; in this teaching example, the weights are hand-picked so their effects are easy to inspect.

The example uses two prompts that differ by one word:

- `the river bank` → predict **floods**
- `the money bank` → predict **lends**

The token `bank` starts with the same embedding in both prompts. Attention incorporates the earlier words, creating a context-dependent representation that leads to a different next-token prediction.

## From prompt to next token

The model repeats this process for each generated token:

1. **Tokenize:** split the prompt into tokens and map them to vocabulary IDs.
2. **Embed:** look up a vector for each token.
3. **Add position:** combine each embedding with a positional vector so the model can distinguish word order.
4. **Build queries, keys, and values:** learned linear projections let each token ask for relevant context, advertise what it contains, and pass information onward.
5. **Attend causally:** calculate scaled query–key scores, mask future tokens, apply softmax, and use the resulting weights to combine value vectors. The causal mask ensures a position can only use itself and earlier tokens.
6. **Use multiple heads:** separate attention heads can focus on different relationships; their results are combined into one representation.
7. **Transform and refine:** residual connections and normalization preserve and stabilize information, while a feed-forward network transforms each token representation independently.
8. **Predict:** use the final position's representation to produce a logit for each vocabulary item; softmax converts logits into next-token probabilities.
9. **Choose and repeat:** greedy decoding picks the highest-probability token. Temperature changes how sharp the distribution is; top-k and top-p restrict the candidate set before sampling. Append the chosen token and repeat until the end-of-sequence token.

## Why it is decoder-only

A full Transformer can contain an encoder and a decoder. GPT-style models use the decoder stack: masked self-attention lets the model generate autoregressively without seeing future tokens. There is no separate encoder or encoder–decoder cross-attention in this demo.

## Scale of the example

| Setting | This demo | Llama 3.1 8B (approx.) |
| --- | ---: | ---: |
| Vocabulary | 7 tokens | 128,000 tokens |
| Embedding width | 4 | 4,096 |
| Attention heads | 2, width 2 each | 32, width 128 each |
| Transformer blocks | 1 | 32 |
| Parameters | 136 hand-picked values | About 8.03 billion learned values |

These figures are illustrative. The demo uses a single block, a tiny vocabulary, and manually chosen values; real language models use many layers and are trained on large datasets.

## Files

- [`file_demo_decoder_only_transformer.md`](file_demo_decoder_only_transformer.md) — notes on temperature, top-k, and top-p decoding.
- [`output.json`](output.json) — currently empty (`[]`).

## Source

This write-up summarizes the supplied *Machine Learning Applications: Generative AI & Large Language Models* lecture slides. The Transformer architecture was introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
