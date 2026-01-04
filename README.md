# T.H.R.E.A.D.

> **T**emporal **H**euristic **R**apid **E**valuation **A**nd **D**ynamics

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg) ![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go) ![Loom](https://img.shields.io/badge/Loom-Native-purple)

**T.H.R.E.A.D.** is the advanced telemetry and benchmarking suite for the [Loom](https://github.com/openfluke/loom) AI framework.

Unlike traditional benchmarks (MMLU, HumanEval) that measure the *destination* (static accuracy), **T.H.R.E.A.D.** measures the *trajectory* of an algorithm during its first minute of life. It acts as a **dynamometer for learning algorithms**, stressing them with non-stationary data streams to quantify Plasticity, Stability, and Memory in real-time.

---

## 🕰️ The Philosophy

**"The First Minute is Everything."**

In Edge AI, Robotics, and Real-Time Systems, you do not have the luxury of offline batch training. Models must adapt *now*. T.H.R.E.A.D. treats the training process itself as the product. It answers:

* **Wake-up Time:** How many milliseconds to go from random noise to usable predictions?
* **Plasticity:** When physics change (e.g., frequency shift), does the model adapt or crash?
* **Memory:** When the environment returns to a previous state, did the model remember it?
* **Safety:** Is the model "sort of wrong" (10% error) or "hallucinating" (>100% error)?

---

## 📐 The Formulas (Component Metrics)

T.H.R.E.A.D. calculates six specific dimensions of learning health before aggregating them.

### 1. Throughput Score ($S_{tput}$)
Raw speed is important, but with diminishing returns. We use a base-10 logarithm so that 100k tok/sec isn't weighted 10x higher than 10k tok/sec (which is already sufficient for real-time).
$$S_{tput} = \log_{10}(\text{TokensPerSecond})$$

### 2. Stability Index ($I_{stab}$)
Measures how smooth the learning curve is. High variance (thrashing) is penalized.
$$I_{stab} = \max(0, 100 - \sigma_{acc})$$
*Where $\sigma_{acc}$ is the standard deviation of accuracy across all time windows.*

### 3. Consistency Rate ($R_{cons}$)
A reliability metric. It asks: "Can I trust this model right now?"
$$R_{cons} = \frac{\text{Windows where Accuracy} > \text{Threshold}}{\text{Total Windows}}$$

### 4. Plasticity Quotient ($Q_{plast}$)
Measures adaptation velocity. It calculates the "Recovery Time" ($t_{rec}$) required to return to 50% accuracy after a major task switch (e.g., 1.0Hz -> 2.0Hz).
$$Q_{plast} = \frac{1000}{t_{rec} \text{ (ms)} + \epsilon}$$
*(Example: Recovering in 200ms yields a score of 5.0. Recovering in 2000ms yields 0.5)*

### 5. Memory Delta ($\Delta_{mem}$)
Measures Catastrophic Forgetting. It compares performance on the *first visit* to a task vs. a *return visit* after interference.
$$\Delta_{mem} = \text{Acc}_{\text{visit 2}} - \text{Acc}_{\text{visit 1}}$$
*(Positive scores indicate true learning/compression. Negative scores indicate overwriting.)*

### 6. Precision Score ($S_{prec}$)
Derived from `DeviationMetrics`, this penalizes "Hallucinations" (errors > 100%) much more heavily than "Inaccuracies" (errors < 10%).
$$S_{prec} = \frac{1}{N} \sum_{i=1}^{N} \max(0, 100 - \text{Deviation}_i)$$

---

## 🏆 The T.H.R.E.A.D. Composite Score

The final score is a single integer that balances **Speed** (Log-Throughput) against **Intelligence** (Stability, Plasticity, Memory).

$$\text{Score} = \left( S_{tput} \times I_{stab} \times R_{cons} \right) \times \left( 1 + \frac{Q_{plast}}{10} + \frac{\Delta_{mem}}{100} \right)$$

### Why this formula?
1.  **The Engine ($S_{tput} \times I_{stab} \times R_{cons}$):**
    * A fast but unstable model gets a low score.
    * A stable but slow model gets a moderate score.
    * A fast AND stable model gets a high base score.
2.  **The Multipliers (Plasticity & Memory):**
    * Models that adapt instantly ($Q_{plast}$) get a massive bonus multiplier.
    * Models that remember the past ($\Delta_{mem}$) get a persistence bonus.

---

## 🔬 The Museum (Benchmarks)

### Test 41: Sine Wave Frequency Adaptation ("The Idiot Test")
A brutal test of plasticity. The model must predict the next value of a sine wave, but the frequency switches every 150ms (`1.0x` → `2.0x` → `3.0x` → `1.0x`).

* **Goal:** Adapt to the new physics within <500ms.
* **Challenge:** Most gradient descent methods fail to adapt fast enough or forget the previous frequency immediately.
* **Winner:** `StepTweenChain` (Loom's geometric update with chain rule).

*(Coming Soon: The Shifting Class MNIST, The Windy Pole Control, The Dynamic Cipher)*

---

## 🚀 Usage

Run the suite locally to verify Edge Readiness:

```bash
# Clone the repository
git clone [https://github.com/openfluke/thread.git](https://github.com/openfluke/thread.git)
cd thread

# Run the Sine Wave Adaptation Benchmark (Test 41)
go run oldexample/test41_sine_adaptation_60s_idiot.go

```

**Sample Output:**

```text
╔═════════════════════════════════════════════════════════════════════════════════════╗
║   🌊 TEST 41: SINE WAVE ADAPTATION BENCHMARK                                        ║
║   TRAINING: Cycle Sin(1x)→Sin(2x)→Sin(3x)→Sin(1x) [IDIOT TEST]                      ║
╚═════════════════════════════════════════════════════════════════════════════════════╝

🚀 [StepTweenChain] Starting...
✅ [StepTweenChain] Done | Acc: 57.4% | Stab: 83% | Cons: 79% | Tput: 10504 | Score: 684

🏆 WINNER: StepTweenChain

```

---

## 📜 License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

```

