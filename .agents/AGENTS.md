The `github.com/gomlx/gomlx.github.io` repo is a Hugo-based documentation for the GoMLX project.

The source contents are in `./content` and the theme and custom components are in `./themes/gomlx`.

It uses Hugo to build the static site in `./public`.


Some of the `./content/docs` files are copied from the other repos under `github.com/gomlx/...`, and it
uses the tool `./cmd/sync_docs` to update those contents.

Most of Go snippets in the `.md` files under `./contents` are copied from examples under
`github.com/gomlx/gomlx/examples/gomlx.github.io/...`, using the tool `./cmd/sync_code`.
This makes sure the snippet code is tested and doesn't go stale. 

It works as follows:

- The `.md` file includes special comments like: `<!-- sync_code: file=core-concepts/tensors/main.go tag=create -->`
- The example `.go` file will have comments tagging the lines that should be copied over to the `.md` file:
  - A trailing `//md:<tag>` indicates that this line should go into the tag `<tag>` snippet.
  - Blocks between `//md_start:<tag1>,<tag2>,...` and `//md_end:` will go into all the listed tags.
- You can run `./cmd/sync_code` (or `make sync_code`) to update the snippets accoding to the code.
- The output of the example can also be synced into the `.md` files, using a marker line like:
  `<!-- sync_code: file=core-concepts/tensors/main.go output_tag=create -->`
  - The same `./cmd/sync_code` will execute the example and paste the output contents marked with the tag.
  - In the `.go` example, output lines `md:<tag1>,<tag2>,...\n` to mark the start of the output for the
    corresponding tags.

See example of this scheme in `./content/docs/core-concepts.md` and 
`github.com/gomlx/gomlx/examples/gomlx.github.io/core-concepts/tensors/main.go`

