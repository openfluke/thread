package main

import (
	"embed"
	"encoding/json"
	"html/template"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
)

//go:embed templates/*
var templatesFS embed.FS

//go:embed static/*
var staticFS embed.FS

// ═══════════════════════════════════════════════════════════════════════════════
// AI DOMAIN TAXONOMY
// ═══════════════════════════════════════════════════════════════════════════════

type AIDomain struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Icon        string `json:"icon"`
	Description string `json:"description"`
	Color       string `json:"color"`
}

type DifficultyLevel struct {
	Level       int    `json:"level"`
	Name        string `json:"name"`
	Description string `json:"description"`
	ModelSize   string `json:"modelSize"`
}

var AIDomains = []AIDomain{
	{ID: "classification", Name: "Classification", Icon: "🏷️", Description: "Assigning discrete labels to inputs", Color: "#3b82f6"},
	{ID: "regression", Name: "Regression", Icon: "📈", Description: "Predicting continuous numerical values", Color: "#10b981"},
	{ID: "generation", Name: "Generation", Icon: "✨", Description: "Creating new data from learned distributions", Color: "#8b5cf6"},
	{ID: "sequence", Name: "Sequence", Icon: "🔗", Description: "Processing time-series and sequential data", Color: "#f59e0b"},
	{ID: "reinforcement", Name: "Reinforcement", Icon: "🎮", Description: "Learning optimal actions from environment feedback", Color: "#ef4444"},
	{ID: "anomaly", Name: "Anomaly Detection", Icon: "🔍", Description: "Identifying outliers and unusual patterns", Color: "#ec4899"},
	{ID: "clustering", Name: "Clustering", Icon: "🎯", Description: "Unsupervised grouping of similar data", Color: "#14b8a6"},
	{ID: "embedding", Name: "Embedding", Icon: "🧬", Description: "Learning dense vector representations", Color: "#6366f1"},
	{ID: "optimization", Name: "Optimization", Icon: "⚡", Description: "Finding optimal hyperparameters and architectures", Color: "#f97316"},
	{ID: "metalearning", Name: "Meta-Learning", Icon: "🧠", Description: "Learning to learn across tasks", Color: "#a855f7"},
}

var DifficultyLevels = []DifficultyLevel{
	{Level: 1, Name: "Trivial", Description: "XOR, AND gates, 2D separable", ModelSize: "< 100 params"},
	{Level: 2, Name: "Toy", Description: "Simple sine waves, tiny datasets", ModelSize: "100-1K params"},
	{Level: 3, Name: "Basic", Description: "Iris, small tabular problems", ModelSize: "1K-10K params"},
	{Level: 4, Name: "Standard", Description: "MNIST digits, basic sequences", ModelSize: "10K-100K params"},
	{Level: 5, Name: "Moderate", Description: "Fashion-MNIST, medium complexity", ModelSize: "100K-500K params"},
	{Level: 6, Name: "Challenging", Description: "CIFAR-10, longer sequences", ModelSize: "500K-1M params"},
	{Level: 7, Name: "Advanced", Description: "CIFAR-100, complex patterns", ModelSize: "1M-10M params"},
	{Level: 8, Name: "Expert", Description: "ImageNet subsets, transformers", ModelSize: "10M-100M params"},
	{Level: 9, Name: "Production", Description: "Full ImageNet, large models", ModelSize: "100M-1B params"},
	{Level: 10, Name: "Frontier", Description: "State-of-the-art, massive scale", ModelSize: "> 1B params"},
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

type ExperimentConfig struct {
	Layers      []string `json:"layers"`
	NumericType string   `json:"numericType"`
	TrainMode   string   `json:"trainMode"`
	BatchSize   int      `json:"batchSize"`
}

type TimeWindow struct {
	TimeMs   int     `json:"timeMs"`
	Accuracy float64 `json:"accuracy"`
	Outputs  int     `json:"outputs"`
}

type ExperimentResult struct {
	Windows          []TimeWindow `json:"windows"`
	Score            float64      `json:"score"`
	AvgAccuracy      float64      `json:"avgAccuracy"`
	Stability        float64      `json:"stability"`
	Consistency      float64      `json:"consistency"`
	ThroughputPerSec float64      `json:"throughputPerSec"`
}

type Experiment struct {
	ID          string           `json:"id"`
	Name        string           `json:"name"`
	Domain      string           `json:"domain"`
	Level       int              `json:"level"`
	Dataset     string           `json:"dataset"`
	Description string           `json:"description"`
	Config      ExperimentConfig `json:"config"`
	Results     ExperimentResult `json:"results"`
	ModelPath   string           `json:"modelPath,omitempty"`
	Status      string           `json:"status"` // pending, running, complete
	CreatedAt   string           `json:"createdAt"`
}

type ExperimentStore struct {
	Experiments []Experiment `json:"experiments"`
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA LAYER
// ═══════════════════════════════════════════════════════════════════════════════

var experiments = &ExperimentStore{Experiments: []Experiment{}}

func loadExperiments() {
	data, err := os.ReadFile("data/experiments.json")
	if err != nil {
		log.Println("📁 No existing experiments found, starting fresh")
		// Create sample experiments for demo
		experiments.Experiments = createSampleExperiments()
		return
	}
	if err := json.Unmarshal(data, experiments); err != nil {
		log.Println("⚠️ Failed to parse experiments.json:", err)
		experiments.Experiments = createSampleExperiments()
	}
	log.Printf("✅ Loaded %d experiments from data/experiments.json\n", len(experiments.Experiments))
}

func saveExperiments() error {
	os.MkdirAll("data", 0755)
	data, err := json.MarshalIndent(experiments, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile("data/experiments.json", data, 0644)
}

func createSampleExperiments() []Experiment {
	return []Experiment{
		{
			ID:          "regression-sine-level2-steptweenchain",
			Name:        "Sine Wave Adaptation",
			Domain:      "regression",
			Level:       2,
			Dataset:     "Synthetic Sine",
			Description: "Rapid frequency switching benchmark (Test 41)",
			Config: ExperimentConfig{
				Layers:      []string{"Dense(11,128)", "Dense(128,128)", "Dense(128,1)"},
				NumericType: "float32",
				TrainMode:   "StepTweenChain",
				BatchSize:   1,
			},
			Results: ExperimentResult{
				Score:            684,
				AvgAccuracy:      57.4,
				Stability:        83,
				Consistency:      79,
				ThroughputPerSec: 10504,
			},
			Status:    "complete",
			CreatedAt: "2026-01-05T09:00:00Z",
		},
		{
			ID:          "classification-mnist-level4-normalbp",
			Name:        "MNIST Digit Classification",
			Domain:      "classification",
			Level:       4,
			Dataset:     "MNIST",
			Description: "Standard digit recognition benchmark",
			Config: ExperimentConfig{
				Layers:      []string{"Dense(784,256)", "Dense(256,128)", "Dense(128,10)"},
				NumericType: "float32",
				TrainMode:   "NormalBP",
				BatchSize:   32,
			},
			Results: ExperimentResult{
				Score:            520,
				AvgAccuracy:      94.2,
				Stability:        92,
				Consistency:      88,
				ThroughputPerSec: 8500,
			},
			Status:    "complete",
			CreatedAt: "2026-01-05T08:30:00Z",
		},
		{
			ID:          "sequence-memory-level3-lstm",
			Name:        "Sequence Memory Task",
			Domain:      "sequence",
			Level:       3,
			Dataset:     "Synthetic Patterns",
			Description: "Testing memory retention over long sequences",
			Config: ExperimentConfig{
				Layers:      []string{"LSTM(32,64)", "Dense(64,10)"},
				NumericType: "float32",
				TrainMode:   "TweenChain",
				BatchSize:   16,
			},
			Results: ExperimentResult{
				Score:            445,
				AvgAccuracy:      78.5,
				Stability:        76,
				Consistency:      82,
				ThroughputPerSec: 5200,
			},
			Status:    "complete",
			CreatedAt: "2026-01-05T08:00:00Z",
		},
	}
}

func getExperimentsByDomain(domain string) []Experiment {
	var result []Experiment
	for _, exp := range experiments.Experiments {
		if exp.Domain == domain {
			result = append(result, exp)
		}
	}
	return result
}

func getExperimentsByDomainAndLevel(domain string, level int) []Experiment {
	var result []Experiment
	for _, exp := range experiments.Experiments {
		if exp.Domain == domain && exp.Level == level {
			result = append(result, exp)
		}
	}
	return result
}

func getExperimentByID(id string) *Experiment {
	for _, exp := range experiments.Experiments {
		if exp.ID == id {
			return &exp
		}
	}
	return nil
}

func getDomainByID(id string) *AIDomain {
	for _, d := range AIDomains {
		if d.ID == id {
			return &d
		}
	}
	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPLATE RENDERING (SSR)
// ═══════════════════════════════════════════════════════════════════════════════

type PageData struct {
	Title           string
	Description     string
	Canonical       string
	Domains         []AIDomain
	Levels          []DifficultyLevel
	Experiments     []Experiment
	Experiment      *Experiment
	CurrentDomain   *AIDomain
	CurrentLevel    *DifficultyLevel
	ExperimentCount int
	DomainCounts    map[string]int
}

func renderTemplate(c *fiber.Ctx, name string, data PageData) error {
	tmpl, err := template.ParseFS(templatesFS, "templates/base.html", "templates/"+name+".html")
	if err != nil {
		log.Println("Template error:", err)
		return c.Status(500).SendString("Template error: " + err.Error())
	}

	c.Set("Content-Type", "text/html; charset=utf-8")
	return tmpl.ExecuteTemplate(c, "base", data)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROUTES
// ═══════════════════════════════════════════════════════════════════════════════

func main() {
	log.Println("🌊 T.H.R.E.A.D. - Temporal Heuristic Rapid Evaluation And Dynamics")
	log.Println("═══════════════════════════════════════════════════════════════════")

	// Load existing experiments
	loadExperiments()

	app := fiber.New(fiber.Config{
		AppName: "T.H.R.E.A.D. Benchmark Suite",
	})

	// Middleware
	app.Use(logger.New())

	// Static files
	app.Static("/static", "./static")

	// ─────────────────────────────────────────────────────────────────────────
	// HOME PAGE
	// ─────────────────────────────────────────────────────────────────────────
	app.Get("/", func(c *fiber.Ctx) error {
		domainCounts := make(map[string]int)
		for _, exp := range experiments.Experiments {
			domainCounts[exp.Domain]++
		}

		return renderTemplate(c, "home", PageData{
			Title:           "T.H.R.E.A.D. - AI Benchmark Suite",
			Description:     "Comprehensive AI benchmarking across all domains and difficulty levels",
			Canonical:       "/",
			Domains:         AIDomains,
			Levels:          DifficultyLevels,
			ExperimentCount: len(experiments.Experiments),
			DomainCounts:    domainCounts,
		})
	})

	// ─────────────────────────────────────────────────────────────────────────
	// DOMAIN OVERVIEW
	// ─────────────────────────────────────────────────────────────────────────
	app.Get("/domain/:domain", func(c *fiber.Ctx) error {
		domainID := c.Params("domain")
		domain := getDomainByID(domainID)
		if domain == nil {
			return c.Status(404).SendString("Domain not found")
		}

		exps := getExperimentsByDomain(domainID)

		return renderTemplate(c, "domain", PageData{
			Title:         domain.Name + " - T.H.R.E.A.D.",
			Description:   domain.Description + " benchmarks and experiments",
			Canonical:     "/domain/" + domainID,
			Domains:       AIDomains,
			Levels:        DifficultyLevels,
			Experiments:   exps,
			CurrentDomain: domain,
		})
	})

	// ─────────────────────────────────────────────────────────────────────────
	// DOMAIN + LEVEL
	// ─────────────────────────────────────────────────────────────────────────
	app.Get("/domain/:domain/:level", func(c *fiber.Ctx) error {
		domainID := c.Params("domain")
		levelStr := c.Params("level")
		level, err := strconv.Atoi(levelStr)
		if err != nil || level < 1 || level > 10 {
			return c.Status(400).SendString("Invalid level")
		}

		domain := getDomainByID(domainID)
		if domain == nil {
			return c.Status(404).SendString("Domain not found")
		}

		var currentLevel *DifficultyLevel
		for _, l := range DifficultyLevels {
			if l.Level == level {
				currentLevel = &l
				break
			}
		}

		exps := getExperimentsByDomainAndLevel(domainID, level)

		return renderTemplate(c, "level", PageData{
			Title:         domain.Name + " Level " + levelStr + " - T.H.R.E.A.D.",
			Description:   currentLevel.Name + " " + strings.ToLower(domain.Name) + " experiments",
			Canonical:     "/domain/" + domainID + "/" + levelStr,
			Domains:       AIDomains,
			Levels:        DifficultyLevels,
			Experiments:   exps,
			CurrentDomain: domain,
			CurrentLevel:  currentLevel,
		})
	})

	// ─────────────────────────────────────────────────────────────────────────
	// EXPERIMENT DETAIL
	// ─────────────────────────────────────────────────────────────────────────
	app.Get("/experiment/:id", func(c *fiber.Ctx) error {
		id := c.Params("id")
		exp := getExperimentByID(id)
		if exp == nil {
			return c.Status(404).SendString("Experiment not found")
		}

		domain := getDomainByID(exp.Domain)

		return renderTemplate(c, "experiment", PageData{
			Title:         exp.Name + " - T.H.R.E.A.D.",
			Description:   exp.Description,
			Canonical:     "/experiment/" + id,
			Domains:       AIDomains,
			Levels:        DifficultyLevels,
			Experiment:    exp,
			CurrentDomain: domain,
		})
	})

	// ─────────────────────────────────────────────────────────────────────────
	// COMPARE PAGE
	// ─────────────────────────────────────────────────────────────────────────
	app.Get("/compare", func(c *fiber.Ctx) error {
		return renderTemplate(c, "compare", PageData{
			Title:       "Compare Experiments - T.H.R.E.A.D.",
			Description: "Cross-experiment analysis and comparison tools",
			Canonical:   "/compare",
			Domains:     AIDomains,
			Levels:      DifficultyLevels,
			Experiments: experiments.Experiments,
		})
	})

	// ─────────────────────────────────────────────────────────────────────────
	// DATASET COMPARISON PAGE
	// ─────────────────────────────────────────────────────────────────────────
	app.Get("/datasets", func(c *fiber.Ctx) error {
		return renderTemplate(c, "datasets", PageData{
			Title:       "Dataset Comparison - T.H.R.E.A.D.",
			Description: "Compare experiments grouped by dataset",
			Canonical:   "/datasets",
			Domains:     AIDomains,
			Levels:      DifficultyLevels,
			Experiments: experiments.Experiments,
		})
	})

	app.Get("/dataset/:name", func(c *fiber.Ctx) error {
		datasetName := c.Params("name")
		var exps []Experiment
		for _, exp := range experiments.Experiments {
			if exp.Dataset == datasetName {
				exps = append(exps, exp)
			}
		}

		return renderTemplate(c, "dataset_detail", PageData{
			Title:       datasetName + " Experiments - T.H.R.E.A.D.",
			Description: "Compare all " + datasetName + " experiments across configurations",
			Canonical:   "/dataset/" + datasetName,
			Domains:     AIDomains,
			Levels:      DifficultyLevels,
			Experiments: exps,
		})
	})

	// ─────────────────────────────────────────────────────────────────────────
	// API ENDPOINTS
	// ─────────────────────────────────────────────────────────────────────────
	api := app.Group("/api")

	api.Get("/experiments", func(c *fiber.Ctx) error {
		domain := c.Query("domain")
		levelStr := c.Query("level")

		result := experiments.Experiments

		if domain != "" {
			var filtered []Experiment
			for _, exp := range result {
				if exp.Domain == domain {
					filtered = append(filtered, exp)
				}
			}
			result = filtered
		}

		if levelStr != "" {
			level, _ := strconv.Atoi(levelStr)
			var filtered []Experiment
			for _, exp := range result {
				if exp.Level == level {
					filtered = append(filtered, exp)
				}
			}
			result = filtered
		}

		return c.JSON(result)
	})

	api.Get("/experiments/:id", func(c *fiber.Ctx) error {
		id := c.Params("id")
		exp := getExperimentByID(id)
		if exp == nil {
			return c.Status(404).JSON(fiber.Map{"error": "not found"})
		}
		return c.JSON(exp)
	})

	api.Get("/domains", func(c *fiber.Ctx) error {
		return c.JSON(AIDomains)
	})

	api.Get("/levels", func(c *fiber.Ctx) error {
		return c.JSON(DifficultyLevels)
	})

	api.Get("/stats", func(c *fiber.Ctx) error {
		domainCounts := make(map[string]int)
		levelCounts := make(map[int]int)
		var totalScore float64

		for _, exp := range experiments.Experiments {
			domainCounts[exp.Domain]++
			levelCounts[exp.Level]++
			totalScore += exp.Results.Score
		}

		return c.JSON(fiber.Map{
			"totalExperiments": len(experiments.Experiments),
			"domainCounts":     domainCounts,
			"levelCounts":      levelCounts,
			"avgScore":         totalScore / float64(len(experiments.Experiments)),
		})
	})

	// Leaderboard
	api.Get("/leaderboard", func(c *fiber.Ctx) error {
		sorted := make([]Experiment, len(experiments.Experiments))
		copy(sorted, experiments.Experiments)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].Results.Score > sorted[j].Results.Score
		})
		if len(sorted) > 20 {
			sorted = sorted[:20]
		}
		return c.JSON(sorted)
	})

	// Datasets list
	api.Get("/datasets", func(c *fiber.Ctx) error {
		datasetMap := make(map[string]int)
		for _, exp := range experiments.Experiments {
			datasetMap[exp.Dataset]++
		}

		type DatasetInfo struct {
			Name  string `json:"name"`
			Count int    `json:"count"`
		}
		var datasets []DatasetInfo
		for name, count := range datasetMap {
			datasets = append(datasets, DatasetInfo{Name: name, Count: count})
		}
		sort.Slice(datasets, func(i, j int) bool {
			return datasets[i].Count > datasets[j].Count
		})
		return c.JSON(datasets)
	})

	// Experiments by dataset
	api.Get("/datasets/:name", func(c *fiber.Ctx) error {
		datasetName := c.Params("name")
		var exps []Experiment
		for _, exp := range experiments.Experiments {
			if exp.Dataset == datasetName {
				exps = append(exps, exp)
			}
		}
		return c.JSON(exps)
	})

	// ─────────────────────────────────────────────────────────────────────────
	// START SERVER
	// ─────────────────────────────────────────────────────────────────────────
	log.Println("🚀 Server starting on http://localhost:3000")
	log.Fatal(app.Listen(":3000"))
}
