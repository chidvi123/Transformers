# BERT and GPT Architecture

## 1. BERT (Bidirectional Encoder Representations from Transformers)

### What is BERT?

BERT is an **Encoder-only Transformer** designed mainly for understanding and representing text.

It uses **bidirectional self-attention**, meaning each token can consider both the left and right context.

Example:

`The bank is beside the river`

BERT uses the surrounding context to understand that `bank` refers to a riverbank rather than a financial institution.

### BERT Architecture

**Published architecture diagram:** [GeeksforGeeks – BERT Model](https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/)

**Original research paper:** [BERT – Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### Working of BERT

1. **Tokenization:** Converts the input text into tokens and token IDs.
2. **Token Embeddings:** Converts each token ID into a dense vector.
3. **Positional Embeddings:** Adds information about the position of each token.
4. **Segment Embeddings:** Identifies which sentence each token belongs to (useful for paired-sentence tasks).
5. **Transformer Encoder:** Passes the representations through multiple Encoder layers.

Each Encoder layer contains:
- Bidirectional Multi-Head Self-Attention.
- Residual Connection and Layer Normalization.
- Feed Forward Network.
- Residual Connection and Layer Normalization.

6. **Contextualized Representations:** Produces a contextual representation for every input token.
7. **Task-specific Output Layer:** Uses these representations for downstream tasks.

### BERT for Classification

For sentence classification, the final representation of the special `[CLS]` token is commonly passed to a Linear layer.

**Flow:**

`Input → BERT Encoder → [CLS] Representation → Linear Layer → Softmax → Class Prediction`

Example:

| Class | Probability |
|---|---:|
| Positive | 88% |
| Negative | 12% |

The predicted class is **Positive**.

Softmax is generally used for mutually exclusive classes. For multi-label classification, independent Sigmoid outputs are commonly used.

### BERT Pre-training Objectives

The original BERT model uses two training objectives:

1. **Masked Language Modeling (MLM):** Some input tokens are masked, and the model learns to predict the masked tokens using surrounding context.
2. **Next Sentence Prediction (NSP):** Predicts whether one sentence follows another in the original text.

Note: NSP was used in the original BERT training setup, but not in every later BERT-style model.

### Applications of BERT

- Sentiment Analysis
- Text Classification
- Named Entity Recognition (NER)
- Question Answering
- Search Relevance

---

## 2. GPT (Generative Pre-trained Transformer)

### What is GPT?

GPT is a **Decoder-only Transformer** designed primarily for autoregressive text generation.

It predicts the next token based on the tokens that came before it.

Example:

Input: `I love machine`

Possible next token: `learning`

GPT then uses the extended sequence to predict the next token.

### GPT Architecture

**Published architecture diagram:** [GeeksforGeeks – Introduction to GPT](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-generative-pre-trained-transformer-gpt/)

**Original research paper:** [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Working of GPT

1. **Input Prompt:** Receives the user's input text.
2. **Tokenization:** Converts the text into token IDs. A token can be a complete word, part of a word, or punctuation.
3. **Token Embeddings:** Converts token IDs into dense vectors.
4. **Positional Information:** Adds information about the position of each token.
5. **Transformer Decoder Blocks:** Passes the representations through multiple Decoder layers.

Each Decoder block contains:
- Causal (Masked) Multi-Head Self-Attention.
- Residual Connection and Normalization.
- Feed Forward Network.
- Residual Connection and Normalization.

6. **Linear Layer:** Projects the final hidden representation into logits for every token in the vocabulary.
7. **Softmax:** Converts logits into probabilities for possible next tokens.
8. **Token Selection:** A decoding strategy selects a token based on its probability distribution.
9. **Autoregressive Generation:** Appends the selected token to the existing sequence and repeats the process until generation stops.

### Why Does GPT Use Masked Self-Attention?

GPT must not look at future tokens while predicting the next token.

Example:

`I love machine learning`

| Current Token | Tokens it can attend to |
|---|---|
| I | I |
| love | I, love |
| machine | I, love, machine |
| learning | I, love, machine, learning |

The causal mask prevents a token from attending to future positions.

**Important:** Causal masking is used during both training and inference.

### Linear Layer and Softmax

Suppose:
- Hidden dimension = 512
- Vocabulary size = 50,000

For the final input position:

`(1 × 512) × (512 × 50,000) = 1 × 50,000`

The Linear layer produces 50,000 logits, one for each vocabulary token.

Softmax converts these logits into probabilities.

Example:

| Possible Next Token | Probability |
|---|---:|
| learning | 60% |
| coding | 20% |
| today | 10% |
| Others | 10% |

The decoding strategy selects a **token**, not a probability. The probability represents how likely the model considers that token.

### Autoregressive Generation

Suppose the initial prompt is:

`I am`

| Current Sequence | Predicted Next Token |
|---|---|
| I am | a |
| I am a | boy |
| I am a boy | EOS |

The model keeps extending the sequence one token at a time.

**Important:** GPT does not necessarily start with a `[BOS]` token. In normal chat or completion use, the user's prompt is the initial sequence. Some model formats use special beginning-of-sequence or message-control tokens.

### Training vs Inference

| Training | Inference |
|---|---|
| Learns to predict the next token in training sequences | Generates tokens from a user prompt |
| Uses causal masking | Uses causal masking |
| Compares predictions with actual next tokens | Selects the next token using a decoding strategy |
| Updates weights using loss and backpropagation | Model weights generally remain fixed |

During training, predictions for multiple positions can be calculated in parallel while causal masking prevents access to future tokens.

During inference, tokens are generated sequentially. Practical implementations commonly use a **KV Cache** to reuse previously calculated keys and values instead of recomputing them from scratch.

---

## 3. BERT vs GPT

| Feature | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional Self-Attention | Causal Self-Attention |
| Main Purpose | Text Understanding | Text Generation |
| Original Training Objective | MLM + NSP | Next Token Prediction |
| Typical Output | Contextualized Token Representations | Next Token Probabilities |
| Cross-Attention | Not used | Not used in standard Decoder-only GPT |

## 4. Quick Recap

**BERT:**
- Uses an Encoder-only architecture.
- Uses bidirectional self-attention.
- Mainly used for understanding and classification tasks.
- Classification flow: `[CLS] → Linear → Softmax → Class`.

**GPT:**
- Uses a Decoder-only architecture.
- Uses causal self-attention.
- Mainly used for text generation.
- Generation flow: `Decoder → Linear → Softmax → Token Selection → Append Token → Repeat`.

## References

1. [GeeksforGeeks – BERT Model](https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/)
2. [GeeksforGeeks – Introduction to GPT](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-generative-pre-trained-transformer-gpt/)
3. [Original BERT Research Paper (2018)](https://arxiv.org/abs/1810.04805)
4. [Original GPT Research Paper (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
5. [Original Transformer Research Paper (2017)](https://arxiv.org/abs/1706.03762)

*Architecture diagrams are referenced through the published GeeksforGeeks articles. Refer to the original articles for the figures and their attribution.*