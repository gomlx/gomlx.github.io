package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"k8s.io/klog/v2"
)

var (
	branch  = flag.String("branch", "", "Branch to fetch")
	version = flag.String("version", "", "Version tag to fetch (or 'latest')")
	commit  = flag.String("commit", "", "Commit hash to fetch")
	local   = flag.String("path", "", "Local path to gomlx repository")
)

const repoOwner = "gomlx"
const repoName = "gomlx"
const repoFullName = repoOwner + "/" + repoName

// --- Main ---

func main() {
	mode, ref, err := parseFlags()
	if err != nil {
		klog.Fatalf("Error parsing flags: %v", err)
	}

	hugoTomlPath, err := findHugoToml()
	if err != nil {
		klog.Fatalf("Could not locate hugo.toml: %v", err)
	}
	baseDir := filepath.Dir(hugoTomlPath)
	outDir := filepath.Join(baseDir, "content", "docs")

	// If mode is a remote ref and version="latest", resolve it.
	displayVersion := ref
	if mode == "version" && ref == "latest" {
		klog.V(1).Info("Fetching latest release tag from GitHub...")
		latest, err := fetchLatestReleaseTag()
		if err != nil {
			klog.Fatalf("Failed to fetch latest release: %v", err)
		}
		ref = latest
		displayVersion = latest
	} else if mode == "path" {
		displayVersion = "local"
	}

	klog.V(1).Infof("Updating hugo.toml with version: %s", displayVersion)
	if err := updateHugoToml(hugoTomlPath, displayVersion); err != nil {
		klog.Fatalf("Failed to update hugo.toml: %v", err)
	}

	// 1. Synchronize overview.md from README.md
	klog.V(1).Info("Processing README.md for overview.md...")
	readmeContent, err := getFileContent(mode, ref, "README.md")
	if err != nil {
		klog.Fatalf("Failed to read README.md: %v", err)
	}
	readmeSourceURL := getSourceURL(mode, ref, "README.md")
	var changedCount int
	overviewChanged, err := processOverview(string(readmeContent), readmeSourceURL, outDir)
	if err != nil {
		klog.Fatalf("Failed to process overview.md: %v", err)
	}
	if overviewChanged {
		relPath, err := filepath.Rel(baseDir, filepath.Join(outDir, "overview.md"))
		if err != nil {
			relPath = filepath.Join(outDir, "overview.md")
		}
		fmt.Printf("✅ Updated %s\n", relPath)
		changedCount++
	}

	// 2. Synchronize CHANGELOG.md (which generates changelog.md)
	files := []string{"CHANGELOG.md"}

	if err := os.MkdirAll(outDir, 0755); err != nil {
		klog.Fatalf("Failed to create output dir: %v", err)
	}

	weight := 10

	for _, fname := range files {
		klog.V(1).Infof("Processing %s...", fname)
		content, err := getFileContent(mode, ref, "docs/"+fname)
		if err != nil {
			klog.Fatalf("Failed to read file %s: %v", fname, err)
		}

		sourceURL := getSourceURL(mode, ref, "docs/"+fname)
		changed, err := processFile(fname, string(content), weight, sourceURL, outDir)
		if err != nil {
			klog.Fatalf("Failed to process file %s: %v", fname, err)
		}
		if changed {
			relPath, err := filepath.Rel(baseDir, filepath.Join(outDir, generateSlug(fname)+".md"))
			if err != nil {
				relPath = filepath.Join(outDir, generateSlug(fname)+".md")
			}
			fmt.Printf("✅ Updated %s\n", relPath)
			changedCount++
		}
		weight += 10
	}

	if changedCount == 0 {
		fmt.Println("✅  No updates found.")
	}

	klog.V(1).Infof("Sync complete. %d doc files written to %s", len(files)+1, outDir)
}

// --- Helpers: Configuration & State ---

func parseFlags() (mode string, value string, err error) {
	flag.Parse()
	var count int
	if *branch != "" {
		count++
		mode = "branch"
		value = *branch
	}
	if *version != "" {
		count++
		mode = "version"
		value = *version
	}
	if *commit != "" {
		count++
		mode = "commit"
		value = *commit
	}
	if *local != "" {
		count++
		mode = "path"
		value = *local
	}

	if count > 1 {
		return "", "", fmt.Errorf("-branch, -version, -commit, and -path are mutually exclusive")
	}
	if count == 0 {
		return "version", "latest", nil
	}
	return mode, value, nil
}

func findHugoToml() (string, error) {
	// Look up the directory tree up to 3 levels to find hugo.toml
	paths := []string{"hugo.toml", "../hugo.toml", "../../hugo.toml", "../../../hugo.toml"}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("hugo.toml not found in common parent directories")
}

func updateHugoToml(path, version string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	re := regexp.MustCompile(`(?m)^version\s*=\s*".*"`)
	newContent := re.ReplaceAllString(string(content), fmt.Sprintf(`version = "%s"`, version))
	return os.WriteFile(path, []byte(newContent), 0644)
}

// --- Helpers: Network & IO ---

func doRequest(reqURL string) ([]byte, error) {
	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, err
	}

	// Support GITHUB_TOKEN to bypass rate limits
	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, reqURL)
	}
	return io.ReadAll(resp.Body)
}

func fetchLatestReleaseTag() (string, error) {
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", repoFullName)
	data, err := doRequest(apiURL)
	if err != nil {
		return "", err
	}

	var release struct {
		TagName string `json:"tag_name"`
	}
	if err := json.Unmarshal(data, &release); err != nil {
		return "", err
	}
	return release.TagName, nil
}

func getDocsFiles(mode, value string) ([]string, error) {
	if mode == "path" {
		docsPath := filepath.Join(value, "docs")
		entries, err := os.ReadDir(docsPath)
		if err != nil {
			return nil, err
		}
		var files []string
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") {
				files = append(files, e.Name())
			}
		}
		return files, nil
	}

	// Remote fetch
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/contents/docs?ref=%s", repoFullName, value)
	data, err := doRequest(apiURL)
	if err != nil {
		return nil, err
	}

	var items []struct {
		Name string `json:"name"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	var files []string
	for _, item := range items {
		if item.Type == "file" && strings.HasSuffix(item.Name, ".md") {
			files = append(files, item.Name)
		}
	}
	return files, nil
}

func getFileContent(mode, value, filePath string) ([]byte, error) {
	if mode == "path" {
		return os.ReadFile(filepath.Join(value, filePath))
	}
	url := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s", repoFullName, value, filePath)
	return doRequest(url)
}

func getSourceURL(mode, value, filePath string) string {
	if mode == "path" {
		absPath, _ := filepath.Abs(filepath.Join(value, filePath))
		return fmt.Sprintf("file://%s", absPath)
	}
	return fmt.Sprintf("https://github.com/%s/blob/%s/%s", repoFullName, value, filePath)
}

// --- Helpers: Formatting & Processing ---

func generateSlug(filename string) string {
	base := strings.TrimSuffix(filename, ".md")
	base = strings.ToLower(base)
	reg := regexp.MustCompile(`[^a-z0-9]`)
	return reg.ReplaceAllString(base, "-")
}

func deriveTitle(filename string) string {
	base := strings.TrimSuffix(filename, ".md")
	base = strings.ReplaceAll(base, "-", " ")
	base = strings.ReplaceAll(base, "_", " ")

	// Simple title case implementation
	words := strings.Fields(base)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

func deriveSection(slug string) string {
	switch {
	case matchPrefix(slug, "context", "graph", "tensor", "node", "backend"):
		return "Reference"
	case matchPrefix(slug, "train", "loss", "optim", "metric"):
		return "Training"
	case matchPrefix(slug, "layer", "dense", "conv", "attention"):
		return "Layers"
	case matchPrefix(slug, "example", "mnist", "cifar", "transformer"):
		return "Examples"
	case matchPrefix(slug, "install", "quick", "start", "intro"):
		return "Get started"
	default:
		return "Guides"
	}
}

func matchPrefix(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

func stripH1(content string) string {
	lines := strings.SplitN(content, "\n", 2)
	if len(lines) > 0 && strings.HasPrefix(lines[0], "# ") {
		if len(lines) == 2 {
			return strings.TrimPrefix(lines[1], "\n")
		}
		return ""
	}
	return content
}

func processFile(filename, content string, weight int, sourceURL, outDir string) (bool, error) {
	slug := generateSlug(filename)
	title := deriveTitle(filename)
	section := deriveSection(slug)
	body := stripH1(content)

	frontmatter := fmt.Sprintf(`---
title: "%s"
section: "%s"
weight: %d
source: "%s"
---

`, title, section, weight, sourceURL)

	newContent := []byte(frontmatter + body)
	outPath := filepath.Join(outDir, slug+".md")

	existing, err := os.ReadFile(outPath)
	if err == nil && string(existing) == string(newContent) {
		return false, nil
	}

	err = os.WriteFile(outPath, newContent, 0644)
	if err != nil {
		return false, err
	}
	return true, nil
}

func extractAboutSection(content string) string {
	lines := strings.Split(content, "\n")
	var aboutLines []string
	inAbout := false
	inTipBlock := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "## ") {
			if inAbout {
				// Reached next section
				break
			}
			if strings.Contains(strings.ToLower(trimmed), "about") {
				inAbout = true
				continue
			}
		}

		if inAbout {
			if strings.HasPrefix(trimmed, "> [!") {
				inTipBlock = true
				continue
			}
			if inTipBlock {
				if strings.HasPrefix(trimmed, ">") {
					continue
				}
				inTipBlock = false
			}

			// Replace relative doc image path with hugo /img/ path
			line = strings.ReplaceAll(line, `src="docs/`, `src="/img/`)
			aboutLines = append(aboutLines, line)
		}
	}
	return strings.TrimSpace(strings.Join(aboutLines, "\n"))
}

func extractHighlightsSection(content string) string {
	lines := strings.Split(content, "\n")
	var highlightLines []string
	inHighlights := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "### Highlights") || strings.HasPrefix(trimmed, "## Highlights") {
			inHighlights = true
			highlightLines = append(highlightLines, "## Highlights")
			continue
		}
		if inHighlights {
			if strings.HasPrefix(trimmed, "## ") {
				// Reached next section
				break
			}
			if trimmed == "Some selected highlights:" {
				continue
			}
			highlightLines = append(highlightLines, line)
		}
	}
	return strings.TrimSpace(strings.Join(highlightLines, "\n"))
}

func processOverview(content string, sourceURL, outDir string) (bool, error) {
	about := extractAboutSection(content)
	highlights := extractHighlightsSection(content)

	frontmatter := fmt.Sprintf(`---
title: "Overview"
section: "Get started"
weight: 1
source: "%s"
---

%s

%s
`, sourceURL, about, highlights)

	newContent := []byte(frontmatter)
	outPath := filepath.Join(outDir, "overview.md")

	existing, err := os.ReadFile(outPath)
	if err == nil && string(existing) == string(newContent) {
		return false, nil
	}

	err = os.WriteFile(outPath, newContent, 0644)
	if err != nil {
		return false, err
	}
	return true, nil
}

