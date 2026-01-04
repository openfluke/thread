package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 41: SINE WAVE ADAPTATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Mirrors arc_benchmark.go behavior for SINE WAVE frequency switching:
//   - Run for 10 seconds total
//   - Switch frequency every 2.5 seconds: Sin(1x) → Sin(2x) → Sin(3x) → Sin(4x)
//   - Track PREDICTION ACCURACY % every 50ms window
//   - Calculate: Score = (Throughput × Stability × Consistency) / 100000
//
// TRAINING METHODS (all run in parallel!):
//   - NormalBP: STOPS to batch train (like arc_benchmark)
//   - StepBP: Immediate step-based backprop
//   - Tween: Batch tween (ForwardCPU + periodic TweenStep)
//   - TweenChain: Batch tween with chain rule
//   - StepTween: Step forward + immediate TweenStep
//   - StepTweenChain: Step forward + immediate TweenStep with chain rule
//
// TARGET: < 500ms to adapt after each frequency switch
//

const (
	// Network architecture
	InputSize  = 11 // Sliding window of 10 sine samples + 1 Time feature
	SineWindowSize = 10 
	HiddenSize = 128 // Hidden layer size (Increased for better memory capacity)
	OutputSize = 1  // Predict next sine value

	// Training parameters
	LearningRate      = float32(0.005) // Balanced LR
	InitScale         = float32(0.5)
	AccuracyThreshold = 0.05 // Prediction correct if abs(pred - expected) < threshold

	// Sine wave parameters
	SinePoints     = 100 // Number of points to generate
	SineResolution = 0.1 // Step size for x values

	// Timing - 60 second run
	TestDuration   = 60 * time.Second
	WindowDuration = 10 * time.Millisecond   // 10ms windows for finer tracking
	SwitchInterval = 150 * time.Millisecond // Switch frequency every 150ms (~400 switches)

	// Batch training interval for batch-based methods
	TrainInterval = 2 * time.Millisecond // Faster training updates for rapid switching
)

// TrainingMode enum
type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
	ModeBatchTween
	ModeBatchTweenChain
	ModeBatchStepTween
	ModeBatchStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:            "NormalBP",
	ModeStepBP:              "StepBP",
	ModeTween:               "Tween",
	ModeTweenChain:          "TweenChain",
	ModeStepTween:           "StepTween",
	ModeStepTweenChain:      "StepTweenChain",
	ModeBatchTween:          "BatchTween",
	ModeBatchTweenChain:     "BatchTweenChain",
	ModeBatchStepTween:      "BatchStepTween",
	ModeBatchStepTweenChain: "BatchStepTweenChain",
}

// TimeWindow for 50ms accuracy tracking
type TimeWindow struct {
	TimeMs        int     `json:"timeMs"`
	Outputs       int     `json:"outputs"`
	TotalAccuracy float64 `json:"totalAccuracy"`
	Accuracy      float64 `json:"accuracy"` // Average prediction accuracy %
	FreqSwitches  int     `json:"freqSwitches"`
}

// ModeResult holds per-mode benchmark results
type ModeResult struct {
	Windows          []TimeWindow `json:"windows"`
	TotalOutputs     int          `json:"totalOutputs"`
	TotalFreqSwitch  int          `json:"totalFreqSwitches"`
	TrainTimeSec     float64      `json:"trainTimeSec"`
	AvgTrainAccuracy float64      `json:"avgTrainAccuracy"`
	Stability        float64      `json:"stability"`   // 100 - stddev
	Consistency      float64      `json:"consistency"` // % windows above threshold
	ThroughputPerSec float64      `json:"throughputPerSec"`
	Score            float64      `json:"score"` // T×S×C / 100000
	SegmentAccuracy  []float64    `json:"segmentAccuracy"` // Avg accuracy per frequency segment
}

// BenchmarkResults is the full output
type BenchmarkResults struct {
	Modes       []string               `json:"modes"`
	Results     map[string]*ModeResult `json:"results"`
	Timestamp   string                 `json:"timestamp"`
	Duration    string                 `json:"duration"`
	WindowMs    int                    `json:"windowMs"`
	Frequencies []float64              `json:"frequencies"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🌊 TEST 41: SINE WAVE ADAPTATION BENCHMARK                                        ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   TRAINING: Cycle Sin(1x)→Sin(2x)→Sin(3x)→Sin(1x) [IDIOT TEST] (switch every 150ms) ║")
	fmt.Println("║   Track PREDICTION ACCURACY % every 10ms!                                           ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   → NormalBP: STOPS to batch train (likely fails fast switches)                    ║")
	fmt.Println("║   → StepTweenChain: trains EVERY sample → should adapt rapidly                     ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   Score = (Throughput × Stability × Consistency) / 100000                          ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	// Generate sine wave data for all 4 frequencies
	frequencies := []float64{1.0, 2.0, 3.0, 1.0}
	allInputs := make([][][]float32, len(frequencies))
	allTargets := make([][]float32, len(frequencies))

	for i, freq := range frequencies {
		sineData, timeData := generateSineWave(freq)
		allInputs[i], allTargets[i] = createSamples(sineData, timeData)
	}

	numWindows := int(TestDuration / WindowDuration)
	fmt.Printf("\n📊 Generated %d samples per frequency | %d windows at %dms each\n", SinePoints, numWindows, WindowDuration.Milliseconds())
	fmt.Printf("⏱️  Duration: %s | Frequency switch every %s\n\n", TestDuration, SwitchInterval)

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTween,
		ModeStepTweenChain,
		ModeBatchTween,
		ModeBatchTweenChain,
		ModeBatchStepTween,
		ModeBatchStepTweenChain,
	}

	results := &BenchmarkResults{
		Modes:       make([]string, len(modes)),
		Results:     make(map[string]*ModeResult),
		Timestamp:   time.Now().Format(time.RFC3339),
		Duration:    TestDuration.String(),
		WindowMs:    int(WindowDuration.Milliseconds()),
		Frequencies: frequencies,
	}

	for i, m := range modes {
		results.Modes[i] = modeNames[m]
	}

	// Run benchmarks in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			modeName := modeNames[m]
			fmt.Printf("🚀 [%s] Starting...\n", modeName)

			result := runSineWaveBenchmark(m, allInputs, allTargets, frequencies)

			mu.Lock()
			results.Results[modeName] = result
			mu.Unlock()

			fmt.Printf("✅ [%s] Done | Acc: %.1f%% | Stab: %.0f%% | Cons: %.0f%% | Tput: %.0f | Score: %.0f\n",
				modeName, result.AvgTrainAccuracy, result.Stability, result.Consistency, result.ThroughputPerSec, result.Score)
		}(mode)
	}

	wg.Wait()
	fmt.Println("\n✅ All benchmarks complete!")

	saveResults(results)
	printTimeline(results)
	saveResults(results)
	printTimeline(results)
	printMemoryTest(results) // New memory test table
	printSummary(results)
}

// generateSineWave creates sine wave samples with given frequency multiplier
func generateSineWave(freqMultiplier float64) ([]float64, []float64) {
	data := make([]float64, SinePoints)
	times := make([]float64, SinePoints)
	for i := 0; i < SinePoints; i++ {
		x := float64(i) * SineResolution
		data[i] = math.Sin(freqMultiplier * x)
		times[i] = x // Store phase time
	}
	return data, times
}

// createSamples creates input/target pairs from sine data
func createSamples(data []float64, times []float64) (inputs [][]float32, targets []float32) {
	numSamples := len(data) - SineWindowSize
	inputs = make([][]float32, numSamples)
	targets = make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, InputSize)
		// 1. Fill sine window
		for j := 0; j < SineWindowSize; j++ {
			input[j] = float32((data[i+j] + 1.0) / 2.0)
		}
		// 2. Inject Time Feature (use time of the LAST sample in window)
		// Normalize time? It's 0..10 roughly. Let's dampen it slightly 0.1 scale? 
		// Or just raw. Raw is fine for small values.
		input[SineWindowSize] = float32(times[i+SineWindowSize-1]) 

		inputs[i] = input
		targets[i] = float32((data[i+SineWindowSize] + 1.0) / 2.0)
	}
	return inputs, targets
}

// createNetwork builds a simple Dense network for sine prediction
func createNetwork() *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, 3)
	net.BatchSize = 1

	layer0 := nn.InitDenseLayer(InputSize, HiddenSize, nn.ActivationLeakyReLU)
	scaleWeights(layer0.Kernel, InitScale)
	net.SetLayer(0, 0, 0, layer0)

	layer1 := nn.InitDenseLayer(HiddenSize, HiddenSize, nn.ActivationLeakyReLU)
	scaleWeights(layer1.Kernel, InitScale)
	net.SetLayer(0, 0, 1, layer1)

	layer2 := nn.InitDenseLayer(HiddenSize, OutputSize, nn.ActivationSigmoid)
	scaleWeights(layer2.Kernel, InitScale)
	net.SetLayer(0, 0, 2, layer2)

	return net
}

func scaleWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

// runSineWaveBenchmark runs real-time sine wave frequency switching benchmark
// ReplayItem for Experience Replay
type ReplayItem struct {
	Input  [][]float32
	Target []float32
}

func runSineWaveBenchmark(mode TrainingMode, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) *ModeResult {
	numWindows := int(TestDuration / WindowDuration) // 200 windows at 50ms each
	result := &ModeResult{
		Windows:         make([]TimeWindow, numWindows),
		SegmentAccuracy: make([]float64, 0),
	}

	// Experience Replay Buffer
	replayBuffer := make([]ReplayItem, 0, 2000)
	maxReplaySize := 2000


	// Initialize windows
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	// Create fresh network
	net := createNetwork()
	numLayers := net.TotalLayers()

	// Initialize states based on mode
	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain ||
		mode == ModeBatchTween || mode == ModeBatchTweenChain || mode == ModeBatchStepTween || mode == ModeBatchStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain || mode == ModeBatchTweenChain || mode == ModeBatchStepTweenChain {
			ts.Config.UseChainRule = true
		}
		// ts.Config.LinkBudgetScale = 0.8 // Removed to allow full memory retention
	}

	// Training batch for batch-based methods
	type TrainingSample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	currentFreqIdx := 0
	lastSwitchTime := start
	currentSegmentAcc := 0.0
	currentSegmentCount := 0

	// Gradient Accumulator for Batch Modes
	var gradAccumulator [][]float32
	batchSize := 20
	stepBatchBuffer := make([]TrainingSample, 0, batchSize)

	// Helper to init accumulator
	initAccumulator := func() {
		gradAccumulator = make([][]float32, numLayers+1) 
	}

	// Helper to add gradients from ts.ChainGradients to Accumulator
	addToAccumulator := func() {
		for i, grads := range ts.ChainGradients {
			if grads == nil { continue }
			if gradAccumulator[i] == nil {
				gradAccumulator[i] = make([]float32, len(grads))
			}
			for j, g := range grads {
				gradAccumulator[i][j] += g
			}
		}
	}

	// Helper to apply averaged gradients
	applyAccumulated := func(count int) {
		scale := float32(1.0) / float32(count)
		for i, accGrads := range gradAccumulator {
			if accGrads == nil { continue }
			
			// Copy scaled gradients back to ts.ChainGradients
			if ts.ChainGradients[i] == nil {
				ts.ChainGradients[i] = make([]float32, len(accGrads))
			}
			for j, g := range accGrads {
				ts.ChainGradients[i][j] = g * scale
			}
			
			// Reset accumulator
			for j := range accGrads {
				accGrads[j] = 0
			}
		}
		ts.TweenWeightsChainRule(net, LearningRate)
	}

	// Ensure accumulator is ready if we are using a batch mode
	if ts != nil && (mode == ModeBatchTween || mode == ModeBatchTweenChain || mode == ModeBatchStepTween || mode == ModeBatchStepTweenChain) {
		initAccumulator()
	}

	


	// =========================================================================
	// MAIN TRAINING LOOP: Switch frequency every 2.5 seconds for 10 seconds
	// =========================================================================
	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)

		// Update window (50ms windows)
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		// Check for frequency switch (every 150ms)
		if time.Since(lastSwitchTime) >= SwitchInterval {
			// Record previous segment stats
			if currentSegmentCount > 0 {
				result.SegmentAccuracy = append(result.SegmentAccuracy, currentSegmentAcc/float64(currentSegmentCount))
			} else {
				result.SegmentAccuracy = append(result.SegmentAccuracy, 0)
			}
			currentSegmentAcc = 0
			currentSegmentCount = 0

			currentFreqIdx = (currentFreqIdx + 1) % len(frequencies) // Cycle through frequencies
			lastSwitchTime = time.Now()
			result.TotalFreqSwitch++
			if currentWindow < numWindows {
				result.Windows[currentWindow].FreqSwitches++
			}
		}

		// Get current frequency's data
		inputs := allInputs[currentFreqIdx]
		targets := allTargets[currentFreqIdx]

		// Get sample
		input := inputs[sampleIdx%len(inputs)]
		target := targets[sampleIdx%len(targets)]
		sampleIdx++

		// Forward pass
		var output []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(input)
		case ModeStepBP:
			state.SetInput(input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		case ModeStepTween, ModeStepTweenChain, ModeBatchStepTween, ModeBatchStepTweenChain, ModeBatchTween, ModeBatchTweenChain:
			// Use ts.ForwardPass so TweenState is populated for training
			output = ts.ForwardPass(net, input)
		}

		// Calculate prediction accuracy for this sample
		sampleAcc := 0.0
		if len(output) > 0 {
			pred := output[0]
			if math.Abs(float64(pred-target)) < AccuracyThreshold {
				sampleAcc = 100.0
			}
		}

		// Record to current window
		if currentWindow < numWindows {
			result.Windows[currentWindow].Outputs++
			result.Windows[currentWindow].TotalAccuracy += sampleAcc
			result.TotalOutputs++
		}
		
		// Accumulate for segment stats
		currentSegmentAcc += sampleAcc
		currentSegmentCount++


		// =====================================================================
		// TRAINING - THIS IS WHERE EACH MODE DIFFERS
		// =====================================================================
		switch mode {
		case ModeNormalBP:
			// Batch training - accumulates samples, then PAUSES to train
			trainBatch = append(trainBatch, TrainingSample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				batches := make([]nn.TrainingBatch, len(trainBatch))
				for i, s := range trainBatch {
					batches[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
				}
				net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse"})
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepBP:
			// Immediate step-based backprop
			grad := make([]float32, len(output))
			if len(output) > 0 {
				grad[0] = clipGrad(output[0]-target, 0.5)
			}
			net.StepBackward(state, grad)
			net.ApplyGradients(LearningRate)

		case ModeTween, ModeTweenChain:
			// Batch tween - accumulates samples, trains periodically with regression gradients
			trainBatch = append(trainBatch, TrainingSample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					out := ts.ForwardPass(net, s.Input)
					// Regression gradient: target - output
					outputGrad := make([]float32, len(out))
					if len(out) > 0 {
						outputGrad[0] = s.Target - out[0]
					}
					totalLayers := net.TotalLayers()
					ts.ChainGradients[totalLayers] = outputGrad
					ts.BackwardTargets[totalLayers] = []float32{s.Target}
					ts.TweenWeightsChainRule(net, LearningRate)
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeBatchTween, ModeBatchTweenChain:
			// True Batch Tween - Accumulate gradients, then update once
			trainBatch = append(trainBatch, TrainingSample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					// Use Regression Backprop
                    ts.ForwardPass(net, s.Input)
					ts.BackwardPassRegression(net, []float32{s.Target})
					addToAccumulator()
				}
				// Apply averaged
				applyAccumulated(len(trainBatch))
				
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepTween, ModeStepTweenChain:
			// "Online" Tween adaptation - one step, one sample
			// + Experience Replay

			// 1. Train on CURRENT sample
			outputGrad := make([]float32, len(output))
			if len(output) > 0 {
				outputGrad[0] = target - output[0]
			}
			totalLayers := net.TotalLayers()
			ts.ChainGradients[totalLayers] = outputGrad
			ts.BackwardTargets[totalLayers] = []float32{target}
			ts.TweenWeightsChainRule(net, LearningRate)
			
			// 2. Add to Replay Buffer (Reservoir Sampling / Probabilistic)
			// Strategy: Only keep 1% of incoming samples to stretch the buffer memory horizon
			// 11,000 throughput * 0.01 = 110 inserts/sec. 
			// 2000 size / 110 inserts = ~18 seconds of memory horizon (covers multiple frequency switches!)
			
			if rand.Float32() < 0.01 { 
				if len(replayBuffer) < maxReplaySize {
					replayBuffer = append(replayBuffer, ReplayItem{Input: [][]float32{input}, Target: []float32{target}})
				} else {
					// Overwrite a RANDOM slot, preserving some old memories elsewhere
					idx := rand.Intn(maxReplaySize)
					replayBuffer[idx] = ReplayItem{Input: [][]float32{input}, Target: []float32{target}}
				}
			}

			// 3. Train on REPLAY sample (1:1 ratio)
			if len(replayBuffer) > 100 {
				idx := rand.Intn(len(replayBuffer))
				item := replayBuffer[idx]
				
				rInput := item.Input[0]
				rTarget := item.Target[0]
				
				rOut := ts.ForwardPass(net, rInput)
				rGrad := make([]float32, len(rOut))
				if len(rOut) > 0 {
					rGrad[0] = rTarget - rOut[0]
				}
				ts.ChainGradients[totalLayers] = rGrad
				ts.BackwardTargets[totalLayers] = []float32{rTarget}
				ts.TweenWeightsChainRule(net, LearningRate)
			}

		case ModeBatchStepTween, ModeBatchStepTweenChain:
			// "Online" Mini-Batch - Accumulate gradients over N steps, then update
			// Note: This adds Latency to learning updates
			
			stepBatchBuffer = append(stepBatchBuffer, TrainingSample{Input: input, Target: target})
			
			if len(stepBatchBuffer) >= batchSize {
				for _, s := range stepBatchBuffer {
					// 1. Re-run forward pass (ForwardPass stores activations in ts)
					ts.ForwardPass(net, s.Input)
					
					// 2. Backward Pass (Regression) -> populates ts.ChainGradients
					ts.BackwardPassRegression(net, []float32{s.Target})
					
					// 3. Accumulate
					addToAccumulator()
				}
				applyAccumulated(len(stepBatchBuffer))
				stepBatchBuffer = stepBatchBuffer[:0]
			}
			
			// Optional: Replay could be added here too, but let's test pure Batch Step first.
		}
	}

	// Finalize windows - compute average accuracy per window
	for i := range result.Windows {
		if result.Windows[i].Outputs > 0 {
			result.Windows[i].Accuracy = result.Windows[i].TotalAccuracy / float64(result.Windows[i].Outputs)
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	calculateSummaryMetrics(result)

	return result
}

func calculateSummaryMetrics(result *ModeResult) {
	// Average training accuracy
	sum := 0.0
	for _, w := range result.Windows {
		sum += w.Accuracy
	}
	result.AvgTrainAccuracy = sum / float64(len(result.Windows))

	// Stability: 100 - stddev
	variance := 0.0
	for _, w := range result.Windows {
		diff := w.Accuracy - result.AvgTrainAccuracy
		variance += diff * diff
	}
	variance /= float64(len(result.Windows))
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	// Consistency: % of windows above 50% accuracy (better threshold for sine)
	const consistencyThreshold = 50.0
	aboveThreshold := 0
	for _, w := range result.Windows {
		if w.Accuracy >= consistencyThreshold {
			aboveThreshold++
		}
	}
	result.Consistency = float64(aboveThreshold) / float64(len(result.Windows)) * 100

	// Throughput
	result.ThroughputPerSec = float64(result.TotalOutputs) / result.TrainTimeSec

	// Score = (T × S × C) / 100000
	result.Score = (result.ThroughputPerSec * result.Stability * result.Consistency) / 100000
}

func clipGrad(v, max float32) float32 {
	if v > max {
		return max
	}
	if v < -max {
		return -max
	}
	if math.IsNaN(float64(v)) {
		return 0
	}
	return v
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("test41_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test41_results.json")
}

func printTimeline(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           PREDICTION ACCURACY % (Avg over 6s chunks) — Switching freq every 150ms (400 switches total)                                           ║")
	fmt.Println("║           NormalBP PAUSES to batch train → low throughput | StepTweenChain trains EVERY sample → maintains accuracy                            ║")
	fmt.Println("╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Printf("║ Mode                 ║")

	// Print time headers (showing 10 chunks of 6s = 60s)
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds ", (i+1)*6)
	}
	fmt.Printf("║ Avg   ║ Score      ║\n")
	fmt.Println("╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	// 60s total, 10ms windows => 6000 windows total.
	// 10 columns => 600 windows per column.
	windowsPerCol := 600

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║", modeName)

		// Print accuracy for each column
		for col := 0; col < 10; col++ {
			avgAcc := 0.0
			count := 0
			startW := col * windowsPerCol
			endW := startW + windowsPerCol
			
			for w := startW; w < endW && w < len(r.Windows); w++ {
				avgAcc += r.Windows[w].Accuracy
				count++
			}
			if count > 0 {
				avgAcc /= float64(count)
			}
			fmt.Printf(" %2.0f%%", avgAcc)
		}
		fmt.Printf(" ║ %3.0f%% ║ %10.0f ║\n", r.AvgTrainAccuracy, r.Score)
	}

	fmt.Println("╚══════════════════════╩════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝")
	fmt.Println("                           ↑ Freq switches every 150ms (aggregated view)")
}

func printMemoryTest(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              🧠 MEMORY TEST: RETURN TO BASELINE (1.0Hz)                                        ║")
	fmt.Println("║           Comparing First Visit (Seg 0) vs Return Visit (Seg 3) accuracy                                       ║")
	fmt.Println("╠══════════════════════╦════════════════════╦════════════════════╦═══════════════════════════════════════════════╣")
	fmt.Println("║ Mode                 ║ Visit 1 (Init)     ║ Visit 2 (Return)   ║ Retention Delta                               ║")
	fmt.Println("╠══════════════════════╬════════════════════╬════════════════════╬═══════════════════════════════════════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		v1 := 0.0
		v2 := 0.0
		if len(r.SegmentAccuracy) > 0 {
			v1 = r.SegmentAccuracy[0]
		}
		if len(r.SegmentAccuracy) > 3 {
			v2 = r.SegmentAccuracy[3]
		}

		delta := v2 - v1
		sign := "+"
		if delta < 0 {
			sign = ""
		}

		fmt.Printf("║ %-20s ║ %17.1f%% ║ %17.1f%% ║ %16s%.1f%%                      ║\n", modeName, v1, v2, sign, delta)
	}
	fmt.Println("╚══════════════════════╩════════════════════╩════════════════════╩═══════════════════════════════════════════════╝")
}


func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              🌊 SINE WAVE ADAPTATION SUMMARY 🌊                                                ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                                                ║")
	fmt.Println("║  Mode               │ Avg Accuracy │ Stability │ Consistency │ Throughput  │ Score       │ Freq Switches     ║")
	fmt.Println("║  ───────────────────┼──────────────┼───────────┼─────────────┼─────────────┼─────────────┼───────────────────║")

	bestScore := 0.0
	bestMode := ""

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║  %-18s │  %7.1f%%   │  %6.1f%%  │   %6.1f%%   │  %9.0f  │  %9.0f  │        %d          ║\n",
			modeName, r.AvgTrainAccuracy, r.Stability, r.Consistency, r.ThroughputPerSec, r.Score, r.TotalFreqSwitch)

		if r.Score > bestScore {
			bestScore = r.Score
			bestMode = modeName
		}
	}

	fmt.Println("║                                                                                                                ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  🏆 WINNER: %-18s with Score: %.0f                                                              ║\n", bestMode, bestScore)
	fmt.Println("║                                                                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
