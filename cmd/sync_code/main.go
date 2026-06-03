package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	branch     = flag.String("branch", "", "Branch to fetch")
	version    = flag.String("version", "", "Version tag to fetch (or 'latest')")
	commit     = flag.String("commit", "", "Commit hash to fetch")
	local      = flag.String("path", "", "Local path to gomlx repository")
	singleFile = flag.String("file", "", "Process only this markdown file")
)

const repoOwner = "gomlx"
const repoName = "gomlx"
const repoFullName = repoOwner + "/" + repoName

// --- Main ---

func main() {
	mode, ref, err := parseFlags()
	if err != nil {
		log.Fatalf("Error parsing flags: %v", err)
	}

	// Resolve "latest" version to the actual release tag if needed.
	if mode == "version" && ref == "latest" {
		log.Println("Fetching latest release tag from GitHub...")
		latest, err := fetchLatestReleaseTag()
		if err != nil {
			log.Fatalf("Failed to fetch latest release: %v", err)
		}
		ref = latest
	}

	var files []string
	if *singleFile != "" {
		path := *singleFile
		if _, err := os.Stat(path); err != nil {
			// If the file is not found directly, try looking inside content/docs/ or content/
			cleanPath := filepath.Clean(path)
			baseName := filepath.Base(cleanPath)

			altPath1 := filepath.Join("content", "docs", baseName)
			altPath2 := filepath.Join("content", baseName)

			if _, err1 := os.Stat(altPath1); err1 == nil {
				path = altPath1
			} else if _, err2 := os.Stat(altPath2); err2 == nil {
				path = altPath2
			} else {
				log.Fatalf("Specified file not found: %s (also checked %s and %s)", *singleFile, altPath1, altPath2)
			}
		}
		files = []string{path}
	} else {
		files, err = findMarkdownFiles("content")
		if err != nil {
			log.Fatalf("Failed to find markdown files: %v", err)
		}
	}

	log.Printf("Processing %d markdown file(s) in mode: %s (ref/path: %s)...", len(files), mode, ref)

	var processedCount int
	var errorCount int
	for _, file := range files {
		log.Printf("Syncing code snippets for %s...", file)
		if err := processMarkdownFile(file, mode, ref); err != nil {
			log.Printf("Error processing %s: %v", file, err)
			errorCount++
		} else {
			processedCount++
		}
	}

	log.Printf("Sync complete. %d files successfully processed, %d errors.", processedCount, errorCount)
	if errorCount > 0 {
		os.Exit(1)
	}
}

// --- Flag & Repository Resolution ---

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

func doRequest(reqURL string) ([]byte, error) {
	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, err
	}

	// Support GITHUB_TOKEN to bypass rate limits in CI/development
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

// getFileContentRaw fetches or reads the code file contents directly.
func getFileContentRaw(mode, value, filePath string) ([]byte, error) {
	repoPath := filepath.Join("examples/gomlx.github.io", filePath)
	if mode == "path" {
		fullLocalPath := filepath.Join(value, repoPath)
		return os.ReadFile(fullLocalPath)
	}
	url := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s", repoFullName, value, repoPath)
	return doRequest(url)
}

// getFileContent retrieves the file contents, trying fallback file paths (e.g. replacing "_" with "-") if the primary search path fails.
func getFileContent(mode, value, filePath string) ([]byte, error) {
	content, err := getFileContentRaw(mode, value, filePath)
	if err == nil {
		return content, nil
	}

	// Try fallback by replacing "_" with "-" in the file path.
	fallbackPath := strings.ReplaceAll(filePath, "_", "-")
	if fallbackPath != filePath {
		log.Printf("File %s not found. Trying fallback path: %s", filePath, fallbackPath)
		fallbackContent, fallbackErr := getFileContentRaw(mode, value, fallbackPath)
		if fallbackErr == nil {
			return fallbackContent, nil
		}
	}
	return nil, err
}

// --- Snippet Parsing ---

// parseGoSnippets parses all snippets annotated with //md: or //md_start://md_end: blocks in a Go file.
func parseGoSnippets(goContent string) map[string][]string {
	snippets := make(map[string][]string)
	lines := strings.Split(goContent, "\n")

	// Regular expression to match trailing //md:<tags> comments
	reTrailing := regexp.MustCompile(`\s*//md:([^\s]+)\s*$`)

	var activeTagsStack [][]string
	activeTagsMap := make(map[string]int) // tracks counts of active tags to handle nested active tags cleanly

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// 1. Control Comment: //md_start:tag1,tag2,...
		if strings.HasPrefix(trimmed, "//md_start:") {
			tagsStr := strings.TrimPrefix(trimmed, "//md_start:")
			var tags []string
			for _, t := range strings.Split(tagsStr, ",") {
				t = strings.TrimSpace(t)
				if t != "" {
					tags = append(tags, t)
				}
			}
			activeTagsStack = append(activeTagsStack, tags)
			for _, t := range tags {
				activeTagsMap[t]++
			}
			continue
		}

		// 2. Control Comment: //md_end:
		if strings.HasPrefix(trimmed, "//md_end:") {
			if len(activeTagsStack) > 0 {
				lastIdx := len(activeTagsStack) - 1
				poppedTags := activeTagsStack[lastIdx]
				activeTagsStack = activeTagsStack[:lastIdx]

				for _, t := range poppedTags {
					activeTagsMap[t]--
					if activeTagsMap[t] <= 0 {
						delete(activeTagsMap, t)
					}
				}
			} else {
				activeTagsMap = make(map[string]int)
			}
			continue
		}

		// 3. Regular Code Line (with optional trailing tag comments)
		var trailingTags []string
		processedLine := line
		if loc := reTrailing.FindStringSubmatchIndex(line); loc != nil {
			tagsStr := line[loc[2]:loc[3]]
			for _, t := range strings.Split(tagsStr, ",") {
				t = strings.TrimSpace(t)
				if t != "" {
					trailingTags = append(trailingTags, t)
				}
			}
			// Strip the trailing tag comment from the code line
			processedLine = line[:loc[0]]
		}

		// Gather all unique tags for this line
		lineTags := make(map[string]bool)
		for _, t := range trailingTags {
			lineTags[t] = true
		}
		for t := range activeTagsMap {
			lineTags[t] = true
		}

		// Save the processed line to all active and trailing tags
		for t := range lineTags {
			snippets[t] = append(snippets[t], processedLine)
		}
	}

	return snippets
}

// adjustIndentation dedents a list of lines so that the minimum common indentation is removed.
func adjustIndentation(lines []string) []string {
	if len(lines) == 0 {
		return lines
	}

	var commonPrefix *string
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}

		// Identify leading whitespace
		var idx int
		for idx < len(line) && (line[idx] == ' ' || line[idx] == '\t') {
			idx++
		}
		leading := line[:idx]

		if commonPrefix == nil {
			p := leading
			commonPrefix = &p
		} else {
			common := *commonPrefix
			limit := len(common)
			if len(leading) < limit {
				limit = len(leading)
			}
			matchLen := 0
			for i := 0; i < limit; i++ {
				if common[i] == leading[i] {
					matchLen++
				} else {
					break
				}
			}
			p := common[:matchLen]
			commonPrefix = &p
		}
	}

	if commonPrefix == nil || *commonPrefix == "" {
		return lines
	}

	prefix := *commonPrefix
	result := make([]string, len(lines))
	for i, line := range lines {
		if strings.HasPrefix(line, prefix) {
			result[i] = strings.TrimPrefix(line, prefix)
		} else {
			if strings.TrimSpace(line) == "" {
				result[i] = ""
			} else {
				result[i] = line
			}
		}
	}
	return result
}

// --- Markdown File Processing & Atomicity ---

// findMarkdownFiles recursively lists all markdown files in the specified directory.
func findMarkdownFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// processMarkdownFile reads the markdown file, parses sync_code instructions, fetches/injects snippets,
// and saves the updated content atomically.
func processMarkdownFile(mdPath string, mode, value string) error {
	content, err := os.ReadFile(mdPath)
	if err != nil {
		return fmt.Errorf("failed to read markdown file: %w", err)
	}

	lines := strings.Split(string(content), "\n")
	var newLines []string

	// Match: <!-- sync_code: file=<file_path> tag=<tag> -->
	reComment := regexp.MustCompile(`^\s*<!--\s*sync_code:\s*file=(\S+)\s+tag=(\S+)\s*-->\s*$`)

	// Cache to prevent repetitive requests/reads for the same files in a markdown document.
	fileSnippetsCache := make(map[string]map[string][]string)

	getSnippet := func(filePath, tag string) ([]string, error) {
		snippets, ok := fileSnippetsCache[filePath]
		if !ok {
			goContent, err := getFileContent(mode, value, filePath)
			if err != nil {
				return nil, fmt.Errorf("failed to load Go file: %w", err)
			}
			snippets = parseGoSnippets(string(goContent))
			fileSnippetsCache[filePath] = snippets
		}

		lines, ok := snippets[tag]
		if !ok {
			return nil, fmt.Errorf("tag %q not found in Go file", tag)
		}
		return lines, nil
	}

	i := 0
	modified := false
	for i < len(lines) {
		line := lines[i]
		newLines = append(newLines, line)

		matches := reComment.FindStringSubmatch(line)
		if matches != nil {
			targetFile := matches[1]
			targetTag := matches[2]

			// Fetch the target snippet
			snippetLines, err := getSnippet(targetFile, targetTag)
			if err != nil {
				return fmt.Errorf("error resolving snippet for %s (tag=%s): %w", targetFile, targetTag, err)
			}

			// Adjust the common leading indentation
			adjustedSnippet := adjustIndentation(snippetLines)

			// Look ahead to check if a code block follows the comment.
			nextBlockStartIdx := -1
			for j := i + 1; j < len(lines); j++ {
				trimmed := strings.TrimSpace(lines[j])
				if trimmed == "" {
					continue
				}
				if strings.HasPrefix(trimmed, "```go") || strings.HasPrefix(trimmed, "```") {
					nextBlockStartIdx = j
				}
				break
			}

			if nextBlockStartIdx != -1 {
				// Find the ending tag of the block
				nextBlockEndIdx := -1
				for j := nextBlockStartIdx + 1; j < len(lines); j++ {
					trimmed := strings.TrimSpace(lines[j])
					if strings.HasPrefix(trimmed, "```") {
						nextBlockEndIdx = j
						break
					}
				}

				if nextBlockEndIdx != -1 {
					// Retrieve existing lines to see if they're different
					var existingCodeLines []string
					if nextBlockStartIdx+1 < nextBlockEndIdx {
						existingCodeLines = lines[nextBlockStartIdx+1 : nextBlockEndIdx]
					}

					// Verify if we actually need to change the content
					changed := false
					if len(existingCodeLines) != len(adjustedSnippet) {
						changed = true
					} else {
						for k := range adjustedSnippet {
							if existingCodeLines[k] != adjustedSnippet[k] {
								changed = true
								break
							}
						}
					}

					if changed {
						modified = true
					}

					// Append the fresh code block
					newLines = append(newLines, "```go")
					newLines = append(newLines, adjustedSnippet...)
					newLines = append(newLines, "```")

					// Fast forward index past the old code block
					i = nextBlockEndIdx
				} else {
					// Opening code fence found but no closing fence. We overwrite and append.
					modified = true
					newLines = append(newLines, "```go")
					newLines = append(newLines, adjustedSnippet...)
					newLines = append(newLines, "```")
				}
			} else {
				// No code block following the comment. Append the new code block.
				modified = true
				newLines = append(newLines, "```go")
				newLines = append(newLines, adjustedSnippet...)
				newLines = append(newLines, "```")
			}
		}
		i++
	}

	if !modified {
		log.Printf("No modifications needed for %s", mdPath)
		return nil
	}

	// Atomic file update: write to a temporary file first, then rename.
	dir := filepath.Dir(mdPath)
	tempFile, err := os.CreateTemp(dir, "sync_code_tmp_*.md")
	if err != nil {
		return fmt.Errorf("failed to create temporary file: %w", err)
	}
	tempPath := tempFile.Name()
	defer func() {
		// Clean up the temp file if it still exists (e.g. if writing or renaming failed)
		if _, err := os.Stat(tempPath); err == nil {
			_ = os.Remove(tempPath)
		}
	}()

	outputContent := strings.Join(newLines, "\n")
	if _, err := tempFile.Write([]byte(outputContent)); err != nil {
		tempFile.Close()
		return fmt.Errorf("failed to write to temporary file: %w", err)
	}
	if err := tempFile.Close(); err != nil {
		return fmt.Errorf("failed to close temporary file: %w", err)
	}

	if err := os.Rename(tempPath, mdPath); err != nil {
		return fmt.Errorf("failed to atomically rename temporary file to destination %s: %w", mdPath, err)
	}

	log.Printf("Successfully updated %s", mdPath)
	return nil
}
