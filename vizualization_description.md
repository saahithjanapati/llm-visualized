
## GPT-2 3D Visualization: Scene Summary

This visualization depicts a forward pass through **GPT-2 Small**. The scene is organized so that **sequence position runs along the Z-axis**: from the user’s front-facing view, the **earliest token is farthest away**, and the **latest token is closest**. Each token position is treated as a separate **lane** that persists throughout the visualization.

### 1. Input tokens and embedding stage

At the very bottom of the scene are thin **gray token chips**, one per prompt token, aligned along the Z-axis. Each chip represents an input token and includes its actual **GPT-2 token ID** as a small detail at the base.

These token chips rise into a large **blue tapered prism** representing the **vocabulary embedding matrix**. Its wide base corresponds to the GPT-2 vocabulary size (**50,257**), and its narrow top corresponds to the residual/embedding dimension (**768**). Visually, it is a large-bottom / small-top prism to convey the mapping from a token index to a 768-dimensional embedding vector.

To the right of that is a smaller **green tapered prism** representing the **position embedding matrix**. It works similarly, but the inputs are **position indices** rather than token IDs. Position chips also rise into this matrix, and a 768-dimensional **position embedding vector** emerges from the top.

After both embedding lookups complete, there are two sets of vectors: token embeddings above the vocabulary embedding matrix and position embeddings above the position embedding matrix. The position vectors then move left so they align underneath the token embedding vectors. The two are combined with a **residual-style addition animation**: each vector is represented as a thin rectangular prism broken into sampled color blocks, and the lower vector’s blocks peel upward and collide with the corresponding blocks of the upper vector. This visually represents elementwise addition.

The result is the initial **residual stream**.

### 2. Vector representation and coloring

Vectors throughout the scene are shown as thin rectangular prisms. Their geometry is roughly proportional to the dimensionality they represent, but their colors come from **sampled real GPT-2 activation values**, not from arbitrary decorative coloring.

Because the full vector dimensionality is too large to visualize directly, the scene samples **one value every 64 dimensions** and uses those sampled values to determine color:

* **768-dimensional vectors** → 12 sampled values / 12 visual color blocks
* **3072-dimensional MLP hidden vectors** → 48 sampled values / 48 color blocks
* **64-dimensional Q/K/V vectors** → 1 sampled value / effectively one color per vector

The residual stream uses a multicolored palette derived from sampled activation values. Query, key, and value vectors use constrained color families so their roles remain visually obvious:

* **Query**: blue gradient
* **Key**: green gradient
* **Value / attention output**: orange-red gradient

So the scene is data-driven, but only through a sparse activation sample rather than all dimensions.

### 3. Residual stream and overall transformer stack

Once token and position embeddings are added, the resulting residual vectors continue upward along the **left side** of the scene. This vertical path is the **residual stream**. To its right sit the components of each transformer block.

The model shown is **GPT-2 Small**, so the scene contains:

* **12 transformer layers**
* residual width **768**
* **12 attention heads**
* head dimension **64**
* MLP expansion from **768 → 3072 → 768**

At the very top, after all 12 layers, there is a final layer norm and then an **unembedding** stage, visualized as an inverted version of the vocabulary embedding matrix.

### 4. Structure of each transformer layer

Each transformer layer is visualized as the following sequence:

1. **LayerNorm 1**
2. **Multi-Head Self-Attention**
3. **Attention output projection**
4. **Residual addition**
5. **LayerNorm 2**
6. **MLP up projection**
7. **GELU nonlinearity**
8. **MLP down projection**
9. **Residual addition**

Each layer is placed slightly above the previous one so the outputs from one layer feed directly into the next.

---

## LayerNorm visualization

Both layer norms use the same visual structure.

Each layer norm is shown as a **hollow extruded tube** running across the Z-axis, with one lane per token position. The tube has open ends along Z so the viewer can see through it. For a prompt of length **t**, there are **t lanes**, and each lane has its own entry and exit slit.

When a layer norm begins, the current residual stream vectors are **copied** and these copies move rightward into the layer norm tube. Inside each lane are two small rectangular prisms representing the learnable **scale** and **shift** parameters.

The layer norm animation has three conceptual stages:

### a. Normalize

The copied vector first undergoes a normalization animation. Its sampled blocks deform into a **wave-like pattern**, then settle back into vector form, recolored according to the normalized values.

### b. Scale

The normalized vector moves into the **scale** vector and is visually multiplied by it.

### c. Shift

The scaled result is then added to the **shift** vector using the same blockwise addition animation used earlier for token + position embeddings.

After that, the post-layer-norm vectors exit upward through the top slits of their corresponding lanes.

---

## Multi-Head Self-Attention visualization

### 1. Q/K/V projection setup

For each of the 12 attention heads, there is a grouped set of three matrices placed together:

* **Query matrix** (blue)
* **Key matrix** (green)
* **Value matrix** (orange)

Each matrix is a tapered prism mapping **768 → 64**, so the base is visually much larger than the top. Initially these matrices are gray. When the forward pass through them occurs, they flash, then become dimly tinted in their respective colors so the moving vectors remain easy to see. Later, during concatenation/output stages, they brighten more strongly.

After LayerNorm 1, the post-LN residual vectors are copied and sent to **every head**. As each set reaches a head, it is copied again into separate inputs for the Q, K, and V matrices. So for each token position and each head, separate Q, K, and V inputs are created.

The vectors rise into the Q/K/V matrices, and 64-dimensional query, key, and value vectors emerge. These are smaller than the residual vectors and are colored from sampled activation data within their head-specific color scheme.

### 2. Attention animation

Self-attention is visualized as a staged conveyor rather than as literal matrix multiplication, so the causal structure is easier to read.

First, within each head:

* the fixed **value vectors** rise into a higher row
* the fixed **key vectors** slide horizontally so each key sits directly underneath its matching value
* the **query vectors** stay parked in their own lanes until the head is ready to process them

The heads run in parallel, but inside each head the query vectors are processed one row at a time in lane order. That creates the familiar lower-triangular causal pattern. For a given token position `i`, the query only visits key positions `<= i` during the full-sequence pass.

When a query begins its turn, it moves into the key column and walks across the valid key lanes for that row. At each query-key encounter, a **colored sphere** appears midway between the blue query vector and the green key vector. That sphere represents the **pre-softmax attention score** (the query-key dot product). In the code, the sphere is tagged with the actual layer/head/token indices, and when attention activations are available its color comes from the stored pre-softmax score for that exact query-key pair.

After the query has visited every valid key for that row, the whole set of score spheres rises to the value row and changes to a **grayscale** representation of the **post-softmax attention weights**:

* darker = lower weight
* lighter = higher weight

The query vector then shrinks away. Visually, that is the handoff from "which earlier positions matter?" to "how much of each value vector should be mixed in?"

### 3. Weighted sum of values

After the post-softmax weights are established, the visualization explicitly animates the weighted sum. For each attended key position, a temporary copy of that position's fixed **value vector** is spawned. That duplicate first flies into the corresponding post-softmax score sphere, where it is recolored to represent **value x attention weight**, and then continues into a moving **running-sum vector**.

So the traveling orange/red vector is the accumulator for that head and that query row. It starts effectively hidden, appears on the first merge, and then updates after each additional weighted-value contribution.

When the accumulator has visited every allowed value position, it parks slightly above the value lane for that query token. That parked vector is the head's final **64-dimensional weighted sum** for that token position. Once all rows have been processed, the temporary query travelers, score spheres, and duplicate value vectors disappear, leaving only the attention outputs.

### 4. KV cache mode

The visualization also supports **KV cache mode**.

* During **prefill**, keys and values are computed once and then remain visible in their cached positions instead of disappearing.
* During later **decode** passes, the model only has one live query lane moving through the block, while earlier keys and values are reused from the cache.
* The cached **key vectors** stay horizontally aligned under their corresponding cached **value vectors**, so the stored context remains visible as a stable attention table.

So in decode mode, the system does not replay the full row-by-row attention history for every earlier token. Instead, the newest token attends over a persistent bank of cached keys and values.

---

## Attention output projection and residual addition

After attention is complete, the 12 per-head outputs for each lane are brought back toward the residual-stream side and **concatenated** beneath an **output projection matrix**. This matrix is shown as a rectangular prism with equal-sized base and top, representing **768 → 768**.

For each lane, the 12 head outputs of size 64 line up together to represent the concatenated 768-dimensional attention result. These pass upward through the output projection matrix. After the pass-through, the matrix turns a **pink/purple** color.

The projected attention output then moves left and is added back into the existing residual stream vectors using the same blockwise residual-add animation. This completes the attention half of the transformer block.

---

## Second LayerNorm and MLP block

The new residual vectors continue upward. Copies are made again and sent into **LayerNorm 2**, which uses the same normalize → scale → shift structure as LayerNorm 1.

After LayerNorm 2 comes the MLP.

### 1. MLP up projection

The first MLP matrix is a tapered prism mapping **768 → 3072**, so it has a narrow base and a wide top. After the pass-through it turns **yellow/orange**.

### 2. GELU

Between the two MLP matrices is a **GELU nonlinearity** animation. The vector becomes wavy/distorted briefly, then settles into its transformed state.

### 3. MLP down projection

The second MLP matrix is the reverse taper, mapping **3072 → 768**. After its pass-through it also turns orange-toned.

The resulting 768-dimensional vector then moves left and is added back into the residual stream, completing the MLP half of the transformer block.

This full transformer-block process repeats **12 times**.

---

## Final LayerNorm and unembedding

After layer 12, there is a **final layer norm**. Unlike the earlier layer norms, this one is aligned directly with the residual stream rather than sitting off to the right, because there is no subsequent residual branch to merge back into.

After this final normalization, the vectors enter the **unembedding matrix**, shown as an inverted version of the original vocabulary embedding matrix. It starts gray and turns blue once activated.

At the output, each lane produces a set of **top-k candidates** using **top-k sampling with (k = 40)**. These are shown as rectangular prisms rising at the top of the unembedding stage:

* **40 prisms per lane**
* prism height reflects the assigned probability / relative likelihood
* the selected token is marked with a line and **3D text**

This represents the model’s predicted logits / token-choice stage.

After the output is shown, the UI presents an **advance-to-next-token** state. After a short delay, the scene transitions to the next forward pass, loading the next generated token and repeating the process until the sequence ends.

---

## UI and interaction elements

The visualization also includes supporting UI elements outside the main 3D scene:

* **Right-side component info panel**: when a user clicks a component, a text panel appears describing that component
* **Top-right formula modal**: shows the equation for the currently active computation step
* **Attention selection preview**: for attention-related components, the lower-right / mid-right panel shows an interactive attention grid displaying **pre-softmax and post-softmax attention scores** for the selected head and tokens

---

## Short version of the scene logic

The scene starts with token IDs and position indices, maps them through learned embeddings, adds them into the residual stream, then repeatedly visualizes each GPT-2 Small transformer block as:

**LayerNorm → QKV projection → causal self-attention → head concatenation → output projection → residual add → LayerNorm → MLP up → GELU → MLP down → residual add**

After 12 layers, a final layer norm and unembedding produce top-k token candidates, the chosen token is displayed, and the visualization advances to the next generation step.
