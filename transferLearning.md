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

For an original weight matrix with dimensions \(d_{\text{out}} \times d_{\text{in}}\):

$$A \in \mathbb{R}^{r \times d_{\text{in}}}$$

$$B \in \mathbb{R}^{d_{\text{out}} \times r}$$

Number of trainable parameters:

$$r(d_{\text{in}}+d_{\text{out}})$$

**Example:**

For a 100 × 100 weight matrix and rank 2:

- Original parameters = 10,000.
- A contains 200 parameters.
- B contains 200 parameters.
- Total trainable parameters = 400.

Although BA produces a 100 × 100 matrix, its 10,000 values are generated from only 400 independently trainable parameters.

### Where is LoRA applied?

It can be applied to Transformer attention projections:
- Query (Q)
- Key (K)
- Value (V)
- Output (O)

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

### Conceptual Formula

$$W_{\text{effective}} = D(Q(W)) + \Delta W$$

Where:
- Q(W) = Quantized pretrained weights.
- D = Dequantization for computation.
- ΔW = Trainable LoRA update.

### LoRA vs QLoRA

| LoRA | QLoRA |
|---|---|
| Uses pretrained weights in their loaded precision | Uses a quantized pretrained model |
| Freezes original weights | Freezes quantized base weights |
| Trains LoRA adapters | Trains LoRA adapters |
| Reduces trainable parameters | Also reduces base-model memory |

---

## 7. AdaLoRA (Adaptive Low-Rank Adaptation)

AdaLoRA is an extension of LoRA that dynamically allocates rank across different model layers based on their importance.

### Why do we need it?

In normal LoRA, we generally select a fixed rank before training. However, not every layer needs the same amount of adaptation.

AdaLoRA allocates more rank to important components and less to less-important components.

### Working

1. Start with a predefined rank budget.
2. Evaluate the importance of different low-rank components.
3. Dynamically allocate rank based on importance.
4. Continue training with the adaptive allocation.

### Mathematical Concept

LoRA represents the weight update as:

$$\Delta W = BA$$

AdaLoRA uses an adaptive decomposition based on singular value decomposition:

$$\Delta W = P\Lambda Q^T$$

Where:
- P and Q = Learned low-rank factors.
- Λ = Singular values.
- Rank allocation is adjusted based on component importance.

### LoRA vs AdaLoRA

| LoRA | AdaLoRA |
|---|---|
| Uses a predefined rank | Dynamically allocates rank |
| Rank allocation is generally fixed | Rank allocation changes during training |
| Simpler approach | Adaptive approach |

---

## 8. Overall Comparison

| Technique | Main idea |
|---|---|
| Transfer Learning | Reuse knowledge from a pretrained model |
| Feature Extraction | Freeze pretrained layers and train a new head |
| Full Fine-Tuning | Update all pretrained weights |
| Partial Fine-Tuning | Update selected pretrained layers |
| Gradual Unfreezing | Progressively unfreeze layers |
| Discriminative Fine-Tuning | Use different learning rates for different layers |
| PEFT | Efficiently adapt a model using fewer trainable parameters |
| LoRA | Learn low-rank weight updates |
| QLoRA | Combine LoRA with quantization |
| AdaLoRA | Dynamically allocate LoRA rank |

## 9. Quick Revision

- **Transfer Learning:** Reuse knowledge from a pretrained model.
- **Feature Extraction:** Freeze pretrained layers and train only the new head.
- **Fine-Tuning:** Adapt pretrained weights to a new task.
- **PEFT:** Reduce the number of parameters that need to be trained.
- **LoRA:** Uses two smaller trainable matrices to learn weight updates.
- **QLoRA:** Combines LoRA with quantization to reduce memory usage.
- **AdaLoRA:** Dynamically allocates rank based on importance.

**Final takeaway:** Transfer Learning reuses pretrained knowledge, Fine-Tuning adapts it to a new task, and PEFT techniques such as LoRA, QLoRA and AdaLoRA make that adaptation more efficient.