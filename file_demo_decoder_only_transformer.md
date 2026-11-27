## Step 9 — Choosing the next token: temperature, top-k, top-p

* **Temperature** divides the logits before softmax: < 1 sharpens, > 1 flattens.
* **Top-k** keeps the k most likely tokens, renormalizes, then samples **in proportion** to probability.
* **Top-p** keeps the smallest set whose cumulative probability reaches p, renormalizes, then samples.

The chosen token is appended to the input and steps 1–8 repeat until `<EOS>`.
