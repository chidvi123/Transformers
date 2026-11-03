# Transfer Learning and Fine-Tuning

## 1. Transfer Learning

Transfer Learning is a technique where we reuse knowledge learned by a pretrained model for a new, related task instead of training a model from scratch.

**Example:** A CNN pretrained on ImageNet has already learned edges, shapes and textures. We can reuse it to classify plant diseases.

**Advantages:**
- Requires less training time.
- Works well with smaller datasets.
- Reduces computational requirements.

## 2. Feature Extraction

Feature Extraction uses a pretrained model to extract useful features while keeping its pretrained layers frozen.

**Working:**
1. Load a pretrained model.
2. Remove its original classification head.
3. Add a new classification head.
4. Freeze the pretrained layers.
5. Train only the new classification head.

**Example:** Use a pretrained CNN to extract image features and train a new classifier to identify healthy and diseased leaves.

## 3. Fine-Tuning

Fine-Tuning adapts a pretrained model to a new task by updating its pretrained weights using a task-specific dataset.

### Different Fine-Tuning Techniques

**A. Full Fine-Tuning**
- Unfreeze all pretrained layers.
- Update all model weights using the new dataset.
- Requires more computational resources.

**B. Partial Fine-Tuning**
- Freeze some layers and unfreeze selected layers.
- Example: Freeze the first 5 layers and train the remaining 5 layers along with the new classification head.

**C. Gradual Unfreezing**
- Initially train only the new classification head.
- Gradually unfreeze pretrained layers during training.

**D. Discriminative Fine-Tuning**
- Assign different learning rates to different layers.
- Earlier layers generally receive smaller learning rates, while deeper layers may receive larger learning rates.

### Feature Extraction vs Fine-Tuning

| Feature Extraction | Fine-Tuning |
|---|---|
| Pretrained layers are frozen | Selected or all pretrained layers are trainable |
| Only new layers are trained | Pretrained weights are also updated |
| Less computationally expensive | Requires more training resources |

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)

PEFT is a collection of techniques that adapt pretrained models by updating only a small number of parameters instead of updating the entire model.

**Why do we need PEFT?**

Large language models may contain billions of parameters. Updating all of them requires significant GPU memory and computational resources.

PEFT keeps most original model weights frozen and trains only a small set of additional or selected parameters.

### Popular PEFT Techniques

| Technique | Description |
|---|---|
| LoRA | Adds small trainable matrices to selected model layers |
| QLoRA | Combines LoRA with quantization |
| AdaLoRA | Dynamically allocates LoRA rank based on importance |
| Adapters | Adds small trainable modules between Transformer layers |
| Prompt Tuning | Learns trainable prompt embeddings |
| Prefix Tuning | Learns trainable prefix representations for attention |

---

## 5. LoRA (Low-Rank Adaptation)

LoRA is a PEFT technique that adapts a pretrained model without directly updating its original weight matrices.

Instead, it introduces two smaller trainable matrices, A and B, to learn the required weight update.

### Working

1. Load the pretrained model.
2. Freeze its original weights.
3. Introduce trainable matrices A and B.
4. Train using task-specific data.
5. Update only A and B through backpropagation.

### Important Formula

The effective weight matrix is:

$$W_{\text{new}} = W + \Delta W$$

Where:

$$\Delta W = BA$$

In practical LoRA implementations:

$$W_{\text{new}} = W + \frac{\alpha}{r}BA$$

Where:
- W = Original frozen weight matrix.
- A and B = Trainable low-rank matrices.
- r = Rank.
- α = Scaling parameter.
- ΔW = Learned weight update.

### Understanding Rank

For an original weight matrix with dimensions d_out × d_in:

$$A \in \mathbb{R}^{r \times d_{in}}$$

$$B \in \mathbb{R}^{d_{out} \times r}$$

Number of trainable parameters:

$$r(d_{in}+d_{out})$$

**Example:**

For a 100 × 100 weight matrix and rank 2:

- Original parameters = 10,000.
- A contains 200 parameters.
- B contains 200 parameters.
- Total trainable parameters = 400.

Although BA produces a 100 × 100 matrix, its 10,000 values are generated from only 400 independently trainable parameters.

### Where is LoRA applied?

It can be applied to Transformer weight matrices such as:
- Query (Q)
- Key (K)
- Value (V)
- Output (O)
- Feed-Forward Network (FFN) projections

The exact target matrices depend on the model and configuration.

### Advantages

- Reduces trainable parameters.
- Requires less GPU memory.
- Reduces task-specific storage requirements.
- Allows different task-specific adapters to share a compatible pretrained model.

---

## 6. QLoRA (Quantized Low-Rank Adaptation)

QLoRA combines LoRA with quantization to reduce memory requirements during fine-tuning.

### What is Quantization?

Quantization converts model weights into lower-bit numerical representations to reduce storage and memory usage.

| Precision | Memory per parameter |
|---|---:|
| FP32 | 4 bytes |
| FP16 | 2 bytes |
| INT8 | 1 byte |
| 4-bit | 0.5 bytes |

### Working

1. Load the pretrained model in 4-bit quantized form.
2. Keep the quantized base model frozen.
3. Add LoRA adapters.
4. Train only the LoRA parameters.
5. During computation, the quantized weights are dequantized as needed.

### Conceptual Formula

$$W_{\text{effective}} = D(Q(W)) + \Delta W$$

Where:
- Q(W) = Quantized pretrained weights.
- D = Dequantization for computation.
- ΔW = Trainable LoRA update.

### Important QLoRA Components

**1. NF4 (4-bit NormalFloat)**

A 4-bit quantization format designed for normally distributed neural network weights.

**2. Double Quantization**

Quantizes the quantization constants themselves to further reduce memory usage.

**3. Paged Optimizers**

Help manage optimizer memory spikes by using paged memory management.

**Important:** QLoRA mainly quantizes the frozen pretrained base model. LoRA adapters are generally maintained in higher precision for training. Activations, gradients and optimizer states have their own precision and memory requirements.

### LoRA vs QLoRA

| LoRA | QLoRA |
|---|---|
| Uses pretrained weights in their loaded precision | Uses a quantized pretrained model |
| Freezes original weights | Freezes quantized base weights |
| Trains LoRA adapters | Trains LoRA adapters |
| Reduces trainable parameters | Also reduces base-model memory |

---

## 7. AdaLoRA (Adaptive Low-Rank Adaptation)

AdaLoRA is an extension of LoRA that dynamically allocates rank across different model weight matrices based on their importance during training.

### Why do we need AdaLoRA?

In normal LoRA, we generally select a fixed rank before training.

For example, suppose we select rank 8 for all four Transformer attention matrices:

| Matrix | Initial Rank |
|---|---:|
| Query (Q) | 8 |
| Key (K) | 8 |
| Value (V) | 8 |
| Output (O) | 8 |
| Total | 32 |

However, not every matrix or low-rank component requires the same amount of adaptation.

Some components may contribute more to the new task, while others may contribute very little.

AdaLoRA solves this by dynamically allocating the available rank budget based on component importance.

### A. Mathematical Working

Standard LoRA represents the weight update using two low-rank matrices:

$$\Delta W = BA$$

AdaLoRA uses an SVD-inspired decomposition:

$$\Delta W = P\Lambda Q$$

Where:

- P = Left low-rank factor.
- Λ = Diagonal matrix containing singular-value-like components.
- Q = Right low-rank factor.

Their dimensions are:

$$P \in \mathbb{R}^{d_{out}\times r}$$

$$\Lambda \in \mathbb{R}^{r\times r}$$

$$Q \in \mathbb{R}^{r\times d_{in}}$$

The diagonal components in Λ play an important role because AdaLoRA estimates their importance and can prune less important components.

### B. Importance Estimation

During training, AdaLoRA estimates how important individual low-rank components are to the task.

It considers:
- Magnitude of the components.
- Sensitivity of the training loss to those components, using gradients.
- Moving averages and uncertainty in importance estimates.

A simplified sensitivity-based importance measure is:

$$I_i = \left|\lambda_i \frac{\partial L}{\partial \lambda_i}\right|$$

Where:
- λᵢ = A diagonal component of Λ.
- L = Training loss.
- ∂L/∂λᵢ = Gradient of the loss with respect to that component.

This gives an indication of how much a component matters to the training objective.

AdaLoRA uses importance estimates and their moving averages and uncertainty to make more stable rank allocation decisions.

### C. Adaptive Rank Allocation

The main idea is to retain more important components and prune less important ones.

Consider an illustrative example:

Initially, each selected attention matrix receives rank 8.

| Matrix | Initial Rank |
|---|---:|
| Query | 8 |
| Key | 8 |
| Value | 8 |
| Output | 8 |
| Total | 32 |

After importance estimation and rank allocation, suppose the ranks become:

| Matrix | Adaptive Rank |
|---|---:|
| Query | 10 |
| Key | 4 |
| Value | 10 |
| Output | 6 |
| Total | 30 |

These values are illustrative, not a fixed AdaLoRA allocation.

What happened?

- Query and Value received more rank.
- Key and Output received less rank.
- Less important low-rank components were pruned.
- The available parameter budget was distributed according to importance.

**Important:** AdaLoRA does not simply check which Q, K or V matrix produces the correct answer. It estimates the importance of individual low-rank components using training information such as gradients and component magnitude.

### D. Overall Training Process

1. Initialize the low-rank factors.
2. Perform forward propagation.
3. Calculate the training loss.
4. Perform backpropagation and update trainable parameters.
5. Estimate the importance of low-rank components.
6. Prune less important components according to the rank budget schedule.
7. Continue training with the updated rank allocation.

This process progressively adapts the rank allocation during training.

### LoRA vs AdaLoRA

| LoRA | AdaLoRA |
|---|---|
| Uses a predefined rank | Dynamically allocates rank |
| Rank is generally fixed | Rank allocation changes during training |
| Same configured rank for selected matrices | Different matrices can receive different ranks |
| Simpler parameter allocation | Importance-based parameter allocation |

**Key Takeaway:** LoRA uses a generally fixed rank, whereas AdaLoRA identifies important low-rank components and dynamically allocates the available parameter budget to them.
