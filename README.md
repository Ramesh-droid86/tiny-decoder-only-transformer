# LLMs, a Tiny Transformer, and Time Series

This repository contains concise notes from the supplied lecture slides: how large language models generate text, how a tiny decoder-only Transformer predicts the next token, and how simple time-series forecasts are built and evaluated.

## Large Language Models and Generative AI

Generative AI systems create new content such as text, images, audio, or code. A Large Language Model (LLM) learns patterns in large collections of text and uses them to generate or transform language.

- **Pre-training** learns broad language patterns, context, and general knowledge from large datasets.
- **Fine-tuning** adapts a pre-trained model to a narrower task or domain using more specific examples.
- Common applications include text generation, translation, code generation, summarization, and question answering.
- Prompts and context guide the response, so clear instructions help the model focus on the intended task.

## The Tiny Decoder-Only Transformer

The example makes next-token prediction small enough to follow by hand. Its weights are hand-picked for clarity; real LLM weights are learned during training.

Two prompts differ by one word:

- `the river bank` → predict **floods**
- `the money bank` → predict **lends**

The token `bank` begins with the same embedding in both prompts. Attention incorporates surrounding words, giving it a context-dependent representation and changing the next-token prediction.

### From prompt to next token

The model repeats these steps for each generated token:

1. **Tokenize:** split the prompt into tokens and map them to vocabulary IDs.
2. **Embed:** look up a vector for each token.
3. **Add position:** combine each embedding with a positional vector so word order is represented.
4. **Build queries, keys, and values:** project token vectors to form queries, keys, and values for attention.
5. **Attend causally:** calculate scaled query–key scores, mask future tokens, apply softmax, and combine value vectors. A position can only use itself and earlier tokens.
6. **Use multiple heads:** different attention heads can focus on different relationships; their results are combined.
7. **Transform and refine:** residual connections and normalization refine the representation, while a feed-forward network transforms each token independently.
8. **Predict:** use the final position to produce logits for the vocabulary; softmax converts them into next-token probabilities.
9. **Choose and repeat:** greedy decoding picks the most likely token. Temperature, top-k, and top-p provide alternative sampling controls. Append the chosen token and repeat until the end-of-sequence token.

### Why decoder-only?

A full Transformer may have an encoder and decoder. GPT-style models use a decoder stack with masked self-attention: they generate autoregressively and cannot look at future tokens. This demo has no separate encoder or cross-attention.

### Scale comparison

| Setting | This demo | Llama 3.1 8B (approx.) |
| --- | ---: | ---: |
| Vocabulary | 7 tokens | 128,000 tokens |
| Embedding width | 4 | 4,096 |
| Attention heads | 2, width 2 each | 32, width 128 each |
| Transformer blocks | 1 | 32 |
| Parameters | 136 hand-picked values | About 8.03 billion learned values |

These figures are illustrative. The demo has one block, a tiny vocabulary, and manually chosen values; real LLMs use many layers and are trained on large datasets.

## Time Series Analysis and Forecasting

A time series is a sequence of observations indexed by time, often recorded at regular intervals such as hours, days, or months. Plotting the values against time helps reveal changes and possible trends. Time-series forecasting uses historical observations to estimate future values; the examples in the slides use the series itself rather than additional independent variables.

### Moving-average forecasts

A simple moving average forecasts the next value with the mean of the most recent `w` observations:

```text
forecast = mean(the most recent w observed values)
```

It is easy to calculate and can smooth short-term variation to help show a trend. Its limitations are that it weights all observations in the window equally, can lag behind a changing trend, and does not model seasonality.

A **weighted moving average** gives more influence to recent observations. For a three-period example, weights `3:2:1` give the latest observation the greatest weight; divide the weighted sum by the sum of the weights.

### Validate and update forecasts

- Train on earlier observations and reserve a later period as a test set. Compare forecasts for that period with the actual values.
- Forecast error is `actual - predicted`. **Mean Squared Error (MSE)** is the average squared error; **Root Mean Squared Error (RMSE)** is `sqrt(MSE)`. Lower values indicate smaller errors on the evaluated data.
- Compare forecast windows or methods on held-out data before choosing one. Moving averages can follow broad trends, but short-term movements remain uncertain.
- Forecasts may become less accurate as new data arrives. Re-train or update the model when actual new observations are available.

## Files

- [`file_demo_decoder_only_transformer.md`](file_demo_decoder_only_transformer.md) — notes on temperature, top-k, and top-p decoding.
- [`output.json`](output.json) — currently empty (`[]`).

## Sources

This write-up summarizes the supplied LLM lecture slides (`LLM_v2.pdf`) and Time Series lecture slides (`7 Time Series.pdf`). The Transformer architecture was introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
