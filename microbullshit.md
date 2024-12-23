---
title: Micro Bullshit
---

## 2024-12-23

Shower thoughts today:

I don't think AlphaZero-type search is fit for LLM test-time-exploration (at least a short exploration of the problem):

1. huge amounts of VRAM needed for all the kv cache,

2. PUCT is pretty fair, and I don't think that makes sense where some tokens just don't make sense. We might want an algorithm that minimizes cumulative regret (e.g. Sequential Halving) than an algorithm like PUCT that minimizes cumulative regret, at least in the short term.

maybe [concept LLMs](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/) (however they work) would solve some of the problem because it would let us batch tokens into concepts which we could explore less.
