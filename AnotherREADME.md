# Transformer Architecture – Complete Learning Notes

A detailed conceptual guide to the original Encoder–Decoder Transformer architecture, covering embeddings, positional encoding, attention mechanisms, Encoder and Decoder blocks, training, inference, and autoregressive generation.

---

## Table of Contents

1. Introduction to Transformers
2. Overall Transformer Architecture
3. Tokenization, Embeddings and Positional Encoding
4. Encoder Architecture
5. Self-Attention Mechanism
6. Query, Key and Value
7. Scaled Dot-Product Attention
8. Multi-Head Attention
9. Residual Connections and Layer Normalization
10. Feed Forward Network
11. Decoder Architecture
12. Masked Multi-Head Self-Attention
13. Cross-Attention
14. Encoder vs Decoder
15. Training vs Inference
16. Autoregressive Text Generation
17. Linear Layer and Softmax
18. Complete Translation Example
19. Important Dimensions and Shapes
20. Important Formulas – Quick Revision

---

# 1. Introduction to Transformers

The Transformer is a deep learning architecture introduced in the 2017 research paper **Attention Is All You Need** by Vaswani et al.

It was designed to process sequential data, particularly natural language, using attention mechanisms rather than recurrent processing.

### Why were Transformers introduced?

Before Transformers, architectures like RNNs and LSTMs were commonly used for sequence processing.

However, they had some limitations:

- Sequential processing made training slower.
- Capturing long-range dependencies was difficult.
- Information from earlier tokens could gradually become difficult to retain.
- Parallel processing was limited.

Transformers address these challenges using attention mechanisms.

### Advantages

- **Parallel processing:** Multiple tokens can be processed simultaneously during training.
- **Long-range dependencies:** Attention helps capture relationships between distant tokens.
- **Contextual understanding:** A token's representation incorporates information from other relevant tokens.
- **Scalability:** Multiple Encoder and Decoder layers can be stacked.

---

# 2. Overall Transformer Architecture

The original Transformer follows an **Encoder–Decoder architecture**.

The architecture has two major components:

- **Encoder:** Understands the source sentence.
- **Decoder:** Generates the target sentence using the Encoder's output and previously generated target tokens.

The original paper uses six Encoder layers and six Decoder layers.

### Original Transformer configuration

| Component | Configuration |
|---|---:|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Model dimension (`d_model`) | 512 |
| FFN hidden dimension | 2048 |

## Original Transformer Architecture

![Original Transformer Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20251004125140134507/transformers.webp)

Source: [GeeksforGeeks – Architecture and Working of Transformers](https://www.geeksforgeeks.org/deep-learning/architecture-and-working-of-transformers-in-deep-learning/)

**Important:** The six Encoder and Decoder layers represent network depth, not the number of input or output words.

---

# 3. Tokenization, Embeddings and Positional Encoding

Transformers cannot directly understand text. They require numerical representations.

Consider:

```text
I am a boy
```

## 3.1 Tokenization

The sentence is divided into tokens.

```text
["I", "am", "a", "boy"]
```

Each token is mapped to a numerical token ID using a tokenizer.

## 3.2 Token Embeddings

Each token ID is converted into a numerical vector using an embedding layer.

In the original Transformer:

```text
d_model = 512
```

Therefore, each token receives a 512-dimensional embedding.

For four tokens:

$$
X \in \mathbb{R}^{4 \times 512}
$$

## 3.3 Positional Encoding

Transformers process tokens in parallel and do not inherently understand their order.

For example:

```text
I am a boy
Boy am I a
```

The order changes the meaning.

To provide positional information, positional encoding is added to token embeddings.

$$
Input = TokenEmbedding + PositionalEncoding
$$

The original Transformer uses sinusoidal positional encoding.

For even dimensions:

$$
PE(pos,2i)=
\sin\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

For odd dimensions:

$$
PE(pos,2i+1)=
\cos\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

Where:

- `pos` = Position of the token.
- `i` = Dimension index.
- `d_model` = Model dimension.

After adding positional encoding:

$$
X \in \mathbb{R}^{4 \times 512}
$$

Each token now contains semantic and positional information.

---

# 4. Encoder Architecture

The Encoder consists of six identical layers stacked together in the original Transformer.

Each Encoder layer contains two main sublayers:

1. Multi-Head Self-Attention
2. Feed Forward Network

Each sublayer is followed by a residual connection and Layer Normalization.

### Original Encoder illustration

![Original Transformer Encoder](https://jalammar.github.io/images/t/Transformer_encoder.png)

Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Processing flow

1. Input representations enter Multi-Head Self-Attention.
2. Attention calculates relationships between tokens.
3. Add & Norm combines the original input with the attention output.
4. The result enters the Feed Forward Network.
5. Another Add & Norm is applied.
6. The output is passed to the next Encoder layer.

After six Encoder layers, we obtain contextual representations for every source token.

For example:

```text
Input: I am a boy

Encoder output:
E1 → Representation of I
E2 → Representation of am
E3 → Representation of a
E4 → Representation of boy
```

For four tokens:

$$
EncoderOutput \in \mathbb{R}^{4 \times 512}
$$

**Important:** The Encoder does not produce one single vector for the entire sentence. It produces one contextual representation per input token.

---

# 5. Self-Attention Mechanism

Self-Attention helps a token understand its relationship with other tokens in the same sequence.

Consider:

```text
The boy is playing football because he loves it.
```

To understand the meaning of `he`, the model needs information from `boy`.

Self-Attention helps capture such relationships.

### How it works

1. Every token is transformed into Query, Key and Value vectors.
2. Query and Key are used to calculate attention scores.
3. Softmax converts the scores into attention weights.
4. Attention weights are multiplied by Value vectors.
5. The result is a contextual representation for every token.

---

# 6. Query, Key and Value

Attention uses three important components:

- Query (Q)
- Key (K)
- Value (V)

They are generated from input representations using learned weight matrices.

Let the input representation be `X`.

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

### Query (Q)

Represents what information the current token is looking for.

### Key (K)

Represents the information against which a Query is compared.

### Value (V)

Contains the information that gets combined according to the attention weights.

### Simple analogy

Imagine searching for a book in a library:

- Query: What book am I looking for?
- Key: Information in the catalogue used to identify relevant books.
- Value: The actual information/content associated with the book.

The Query and Key determine relevance, while Value supplies the information to retrieve.

**Important:** Q, K and V are learned transformations of the input representations. They are not simply the original embeddings.

---

# 7. Scaled Dot-Product Attention

Once Q, K and V are generated, attention calculation begins.

The original Transformer uses Scaled Dot-Product Attention.

### Step 1: Calculate attention scores

Multiply Query with the transpose of Key.

$$
Scores = QK^T
$$

This calculates how much attention each Query assigns to each Key.

### Step 2: Scaling

Divide the scores by the square root of the Key dimension.

$$
ScaledScores = \frac{QK^T}{\sqrt{d_k}}
$$

Scaling prevents excessively large dot-product values that can make Softmax gradients very small.

### Step 3: Apply Softmax

Softmax converts the scores into attention weights.

$$
AttentionWeights =
Softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
$$

Each row of the attention weight matrix sums to 1.

### Step 4: Multiply by Value

Multiply the attention weights with the Value matrix.

$$
AttentionOutput =
Softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

This produces a contextual representation for every token.

### Complete formula

$$
\boxed{
Attention(Q,K,V)=
Softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

### Important distinction

| Operation | Meaning |
|---|---|
| `QKᵀ` | Attention scores |
| Softmax of scores | Attention weights |
| Attention weights × V | Attention output |

**Remember:** Attention weights determine how much information to take, and Value contains the information being combined.

---

# 8. Multi-Head Attention

Instead of performing attention only once, the Transformer performs attention multiple times using different learned projections.

This is called Multi-Head Attention.

The original Transformer uses eight attention heads.

### Why multiple heads?

Different heads can learn different relationships between tokens.

For example, in:

```text
The boy is playing football
```

Different heads may learn to focus on:

- Subject relationships.
- Verb relationships.
- Contextual relationships.
- Other linguistic patterns.

These are possible learned patterns, not fixed rules assigned to individual heads.

## Multi-Head Attention calculation

Assume:

```text
Number of tokens = 4
d_model = 512
Number of heads = 8
d_head = 64
```

Each head receives the same input representation but uses its own learned projection matrices.

For each head:

$$
Q_i = XW_{Q_i}
$$

$$
K_i = XW_{K_i}
$$

$$
V_i = XW_{V_i}
$$

Each head produces:

$$
Q_i,K_i,V_i \in \mathbb{R}^{4 \times 64}
$$

### Step 1: Calculate attention scores

$$
Scores_i = Q_iK_i^T
$$

For four tokens:

$$
Scores_i \in \mathbb{R}^{4 \times 4}
$$

Every token's Query is compared with the Keys of the available tokens.

### Step 2: Calculate attention weights

$$
Weights_i =
Softmax\left(
\frac{Q_iK_i^T}{\sqrt{64}}
\right)
$$

$$
Weights_i \in \mathbb{R}^{4 \times 4}
$$

### Step 3: Calculate attention output

$$
Head_i = Weights_iV_i
$$

$$
Head_i \in \mathbb{R}^{4 \times 64}
$$

### Step 4: Concatenate all heads

$$
Concat = Concat(Head_1,Head_2,\ldots,Head_8)
$$

$$
Concat \in \mathbb{R}^{4 \times 512}
$$

### Step 5: Apply output projection

$$
MultiHead(Q,K,V)=Concat(Head_1,\ldots,Head_8)W_O
$$

The final output shape remains:

$$
Output \in \mathbb{R}^{4 \times 512}
$$

### Original Multi-Head Attention illustration

![Original Multi-Head Attention illustration](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

**Important clarification:** The original 512-dimensional embedding is not physically split into eight pieces before attention. Each head uses its own learned projection matrices to produce 64-dimensional Q, K and V vectors.

---

# 9. Residual Connections and Layer Normalization

After Multi-Head Attention, the Transformer uses a residual connection followed by Layer Normalization.

This is called Add & Norm.

### Why add the original input?

Let:

- `X` = Original input.
- `Attention(X)` = Contextual information obtained from attention.

The residual addition is:

$$
Y = X + Attention(X)
$$

The original input contains initial token and positional information, while the attention output contains contextual information gathered from other tokens.

Adding them helps preserve a direct path for the original information.

### Benefits of residual connections

1. Preserve a direct path for original input information.
2. Help gradients flow through deep networks.
3. Make deeper architectures easier to train.

## Layer Normalization

After addition, Layer Normalization is applied.

$$
Output = LayerNorm(X + Attention(X))
$$

Layer Normalization normalizes feature values for each token across its hidden dimensions.

It uses learnable parameters:

- Gamma (γ): Scale.
- Beta (β): Shift.

Its purpose is to stabilize activation distributions and improve training.

### Original residual connection and normalization illustration

![Original Transformer residual connection and layer normalization](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Important distinction

- Residual connection adds the original input.
- Layer Normalization normalizes feature values.
- Softmax converts attention scores into attention weights.

They perform different operations.

The original Transformer uses Post-Layer Normalization, meaning normalization happens after residual addition.

---

# 10. Feed Forward Network

After the first Add & Norm, the representation enters the Feed Forward Network (FFN).

The FFN processes each token representation independently using the same learned network.

Unlike Self-Attention, it does not directly mix information between different token positions.

### Architecture

The original Transformer uses two linear transformations with a ReLU activation in between.

$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$

### Original dimensions

```text
Input dimension  = 512
Hidden dimension = 2048
Output dimension = 512
```

The processing is:

$$
512 \rightarrow 2048 \rightarrow ReLU \rightarrow 512
$$

### Why use FFN?

Attention combines information from different tokens.

The Feed Forward Network further transforms the resulting token representations and learns useful nonlinear patterns.

After the FFN, another residual connection and Layer Normalization are applied.

$$
Output = LayerNorm(X + FFN(X))
$$

Here, `X` represents the input to that FFN sublayer.

### Original Encoder layer illustration

![Original Transformer Encoder block](https://jalammar.github.io/images/t/Transformer_encoder.png)

Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

# 11. Decoder Architecture

The Decoder generates the target sequence.

The original Transformer contains six Decoder layers.

Unlike the Encoder, each Decoder layer contains three main sublayers:

1. Masked Multi-Head Self-Attention.
2. Encoder–Decoder Cross-Attention.
3. Feed Forward Network.

Each sublayer uses Add & Norm.

### Original Decoder illustration

![Original Transformer Decoder](https://jalammar.github.io/images/t/Transformer_decoder.png)

Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

The Decoder has two different attention mechanisms:

- **Masked Self-Attention:** Looks at available target tokens.
- **Cross-Attention:** Uses information from the Encoder's output.

---

# 12. Masked Multi-Head Self-Attention

The Decoder uses Masked Multi-Head Self-Attention to prevent future target tokens from being accessed.

### Why is masking required?

Consider the target sentence:

```text
Je suis un garçon
```

During training, the Decoder receives a shifted version of the target sequence.

```text
Decoder input:   BOS   Je   suis   un
Expected output: Je    suis un     garçon
```

The model must predict the next token at every position.

If the first position can access all future tokens, it could directly see the answer.

This would cause information leakage during training.

### Causal masking

The causal mask prevents a token from attending to future positions.

| Query position | BOS | Je | suis | un |
|---|---:|---:|---:|---:|
| BOS | ✓ | ✗ | ✗ | ✗ |
| Je | ✓ | ✓ | ✗ | ✗ |
| suis | ✓ | ✓ | ✓ | ✗ |
| un | ✓ | ✓ | ✓ | ✓ |

The first token can only attend to itself.

The second token can attend to BOS and Je, but cannot access suis or un.

### Why is this important?

It ensures that the Decoder learns to predict the next token using only previously available target tokens.

During inference, future tokens do not exist yet. However, causal masking remains part of the Decoder architecture.

---

# 13. Cross-Attention

Cross-Attention connects the Encoder and Decoder.

It allows the Decoder to access information from the original source sentence while generating the target sentence.

Consider:

```text
English: I am a boy
French:  Je suis un garçon
```

The Encoder processes the English sentence and produces contextual representations.

These representations are passed to the Cross-Attention layer in the Decoder.

## Where do Q, K and V come from?

This is one of the most important distinctions in the Transformer architecture.

In Cross-Attention:

- Query comes from the Decoder.
- Key comes from the Encoder.
- Value comes from the Encoder.

Let:

```text
Decoder representation = D
Encoder output = E
```

The Cross-Attention layer creates:

$$
Q = DW_Q
$$

$$
K = EW_K
$$

$$
V = EW_V
$$

The Encoder does not directly send separately calculated K and V matrices for Cross-Attention.

It sends its final contextual representations. The Cross-Attention layer transforms those representations into K and V using learned projection matrices.

### Cross-Attention calculation

Once Q, K and V are generated, the same Scaled Dot-Product Attention formula is used.

$$
Attention(Q,K,V)=
Softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

### Original Cross-Attention / Decoder illustration

![Original Transformer Decoder and Cross-Attention illustration](https://jalammar.github.io/images/t/Transformer_decoder.png)

Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Understanding with a translation example

Imagine a human translator.

The translator has the English sentence on one side and is writing the French translation on the other.

Whenever the translator needs information, they look back at the English sentence.

Similarly:

- Query asks what information is needed.
- Key helps determine which source representations are relevant.
- Value provides the information to combine.

For example, when generating the first French token, the Decoder uses Cross-Attention to access relevant information from the Encoder's representation of the English sentence.

**Important:** Cross-Attention does not simply map one English word to one French word. It uses contextual information from the source sentence to help generate the target sequence.

---

# 14. Encoder vs Decoder

| Feature | Encoder | Decoder |
|---|---|---|
| Main purpose | Understand input | Generate output |
| Self-Attention | Normal Multi-Head Self-Attention | Masked Multi-Head Self-Attention |
| Cross-Attention | No | Yes |
| Feed Forward Network | Yes | Yes |
| Add & Norm | Yes | Yes |
| Original number of layers | 6 | 6 |

### Attention differences

**Encoder Self-Attention**

Q, K and V are all generated from Encoder representations.

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

**Decoder Masked Self-Attention**

Q, K and V are generated from Decoder representations, with causal masking applied to prevent access to future target tokens.

**Decoder Cross-Attention**

Q comes from Decoder representations, while K and V come from Encoder representations.

---

# 15. Training vs Inference

Training and inference are two different phases of the Transformer.

Consider the translation task:

```text
English: I am a boy
French:  Je suis un garçon
```

## 15.1 Training Phase

During training, both the source sentence and the correct target sentence are available.

### Encoder input

```text
I am a boy
```

The Encoder processes the entire English sentence and produces contextual representations.

### Decoder input

The correct French sentence is shifted right by one position.

```text
Decoder input:   BOS   Je   suis   un
Expected output: Je    suis un     garçon
```

This approach is called **Teacher Forcing**.

The Decoder uses Masked Self-Attention, Cross-Attention and FFN to generate predictions.

### Loss calculation

Predicted tokens are compared with the correct target tokens using a loss function, commonly cross-entropy loss.

$$
Loss = CrossEntropy(Predictions, TargetTokens)
$$

The loss is used for backpropagation to update the trainable weights of both the Encoder and Decoder.

### Training flow

1. The source sentence is processed by the Encoder.
2. The shifted-right target sentence is provided to the Decoder.
3. The Decoder produces predictions for the target positions.
4. Predictions are compared with the correct target tokens.
5. The loss is used for backpropagation and parameter updates.

## 15.2 Inference Phase

During inference, we only provide the source sentence.

```text
Input: I am a boy
```

The Encoder processes the English sentence.

The Decoder initially receives only the BOS token.

It generates the output one token at a time.

No correct target sentence is supplied, and the model's weights are not updated.

### Main differences

| Training | Inference |
|---|---|
| Source and target sentences are available | Only source sentence is provided |
| Decoder receives shifted correct target tokens | Decoder receives BOS and its own generated tokens |
| Predictions are compared with correct targets | Predictions are used to generate the sequence |
| Loss and backpropagation update weights | No weight updates |
| All target positions can be processed in parallel with causal masking | Generation is autoregressive |

---

# 16. Autoregressive Text Generation

Autoregressive generation means that the model predicts one token at a time and uses previously generated tokens to predict subsequent tokens.

Let's use our translation example.

```text
English: I am a boy
Expected French: Je suis un garçon
```

### Step 1: First prediction

Initially, the Decoder receives:

```text
BOS
```

The token passes through all six Decoder layers.

The processing includes:

1. Masked Self-Attention.
2. Cross-Attention.
3. Feed Forward Network.

The final Decoder representation is sent to Linear + Softmax.

Prediction:

```text
Je
```

### Step 2: Second prediction

The generated token is appended to the previous Decoder input.

```text
BOS Je
```

The sequence passes through all six Decoder layers again.

The last position's representation is used to predict the next token.

Prediction:

```text
suis
```

### Step 3: Third prediction

```text
BOS Je suis
```

Prediction:

```text
un
```

### Step 4: Fourth prediction

```text
BOS Je suis un
```

Prediction:

```text
garçon
```

### Step 5: Termination

The Decoder continues generating tokens until it produces a special token called EOS (End of Sequence).

```text
BOS → Je → suis → un → garçon → EOS
```

Once EOS is generated, the generation process stops.

### Important clarification about Decoder layers

Every generated token passes through all six Decoder layers.

The six layers represent the depth of the network, not the number of tokens it can generate.

| Generation step | Decoder input | Output prediction |
|---|---|---|
| 1 | BOS | Je |
| 2 | BOS Je | suis |
| 3 | BOS Je suis | un |
| 4 | BOS Je suis un | garçon |

The Decoder can generate sentences of different lengths. It is not restricted to six words.

---

# 17. Linear Layer and Softmax

After the final Decoder layer, the resulting representations are passed through a Linear layer.

The Linear layer projects the Decoder's hidden representation into the vocabulary dimension.

For example:

```text
Decoder hidden dimension = 512
Vocabulary size = 30,000
```

The Linear layer transforms:

$$
512 \rightarrow 30000
$$

The resulting values are called logits.

### Softmax

Softmax converts logits into probabilities.

$$
Softmax(z_i)=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Example:

| Token | Probability |
|---|---:|
| Je | 0.75 |
| Tu | 0.10 |
| Il | 0.08 |
| Other tokens | 0.07 |

The model then selects a token according to its decoding strategy.

The selected token is appended to the Decoder input for the next generation step.

**Important:**

- During inference, the final representation at the last available Decoder position is used to predict the next token.
- During training, predictions are produced for all target positions in parallel.

---

# 18. Complete Translation Example

Let's combine everything we have learned.

### Input

```text
English: I am a boy
French:  Je suis un garçon
```

### Original Transformer architecture

![Original Transformer Encoder–Decoder architecture](https://media.geeksforgeeks.org/wp-content/uploads/20251004125140134507/transformers.webp)

Source: [GeeksforGeeks – Architecture and Working of Transformers](https://www.geeksforgeeks.org/deep-learning/architecture-and-working-of-transformers-in-deep-learning/)

### Complete flow in simple words

1. The English sentence enters the Encoder.
2. Tokenization converts the sentence into tokens.
3. Embedding converts token IDs into numerical representations.
4. Positional Encoding adds positional information.
5. The Encoder processes the sentence through six Encoder layers.
6. The Encoder produces contextual representations for every input token.
7. The Decoder initially receives BOS.
8. Masked Self-Attention processes the available target tokens.
9. Cross-Attention uses the Decoder's Query and the Encoder's Key and Value representations.
10. The result passes through the Feed Forward Network.
11. This process continues through all six Decoder layers.
12. The final Decoder representation is passed through Linear and Softmax.
13. The next token is predicted.
14. The generated token is appended to the Decoder input.
15. The process repeats until EOS is generated.

---

# 19. Important Dimensions and Shapes

Let's assume:

```text
Input sentence: I am a boy
Number of tokens: 4
d_model: 512
Number of attention heads: 8
d_head: 64
FFN hidden dimension: 2048
```

## Encoder dimensions

| Operation | Shape |
|---|---|
| Token embeddings | 4 × 512 |
| Positional Encoding | 4 × 512 |
| Final input representation | 4 × 512 |
| Q, K, V for one head | 4 × 64 |
| Attention scores for one head | 4 × 4 |
| Attention weights for one head | 4 × 4 |
| Output of one head | 4 × 64 |
| Concatenated heads | 4 × 512 |
| Output projection | 4 × 512 |
| FFN hidden representation | 4 × 2048 |
| FFN output | 4 × 512 |
| Final Encoder output | 4 × 512 |

## Decoder dimensions during inference

Assuming the same hidden dimension:

| Generation step | Decoder input shape | Final Decoder representation |
|---|---|---|
| 1 | 1 × 512 | 1 × 512 |
| 2 | 2 × 512 | 2 × 512 |
| 3 | 3 × 512 | 3 × 512 |
| 4 | 4 × 512 | 4 × 512 |

During inference, the final position's representation is used for the next-token prediction.

For example, at step 3, the Decoder has three target positions. The third position's final representation is used to predict the fourth token.

**Note:** These dimensions describe the conceptual Transformer computation. Efficient implementations may use batching, caching and other optimizations.

---

# 20. Important Formulas – Quick Revision

### Token representation

$$
Input = TokenEmbedding + PositionalEncoding
$$

### Query, Key and Value

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

### Scaled Dot-Product Attention

$$
Attention(Q,K,V)=
Softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

### Multi-Head Attention

$$
MultiHead(Q,K,V)=
Concat(head_1,\ldots,head_h)W_O
$$

### Feed Forward Network

$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$

### Residual Connection + Layer Normalization

$$
Output=LayerNorm(X+Sublayer(X))
$$

### Cross-Attention

$$
Q=DecoderOutput \times W_Q
$$

$$
K=EncoderOutput \times W_K
$$

$$
V=EncoderOutput \times W_V
$$

### Softmax

$$
Softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

# Final Revision Summary

| Component | Main responsibility |
|---|---|
| Tokenization | Converts text into tokens and token IDs |
| Embedding | Converts token IDs into vectors |
| Positional Encoding | Adds information about token positions |
| Self-Attention | Captures relationships between tokens |
| Query | Represents what information is being searched for |
| Key | Helps determine relevance |
| Value | Supplies information to be combined |
| Softmax | Converts attention scores into attention weights |
| Multi-Head Attention | Performs attention through multiple learned heads |
| Residual Connection | Adds the original input to the sublayer output |
| Layer Normalization | Normalizes feature values |
| Feed Forward Network | Further transforms each token representation |
| Masked Self-Attention | Prevents access to future target tokens |
| Cross-Attention | Connects Decoder representations with Encoder representations |
| Linear Layer | Projects hidden representations to vocabulary logits |
| Softmax | Converts logits into token probabilities |
| EOS | Signals the end of generation |

## Final Takeaway

The Encoder understands the source sentence and generates contextual representations.

The Decoder generates the target sentence by combining information from previously generated target tokens and the Encoder's output.

- Self-Attention captures relationships within a sequence.
- Masked Self-Attention prevents access to future target tokens.
- Cross-Attention connects the source and target sequences.
- The Feed Forward Network further transforms token representations.
- Linear and Softmax produce probabilities for the next token.

The entire process continues until the Decoder produces an End of Sequence token.

---

**Learning status:** Original Encoder–Decoder Transformer architecture — Conceptual understanding completed.

**Next topics:** Consolidate positional encoding, tokenization, embeddings, training objectives and decoding strategies before moving to Encoder-only, Decoder-only and other Transformer variations.