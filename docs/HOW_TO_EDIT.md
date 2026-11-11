# How to edit the experiment log

[← dashboard](README.md)

Everything here is plain Markdown that github.com renders directly. There is no build step and no special syntax.

## Where things are

| path | what it is |
|---|---|
| `README.md` | the dashboard: experiment list, one-line conclusions, headline numbers, key figure |
| `posts/YYYY-MM-DD-short-title.md` | one post per experiment or write-up |
| `assets/img/` | figures (PNG) |
| `_config.yml` | only used if GitHub Pages is turned on later |

## Add a post

1. Create `posts/YYYY-MM-DD-short-title.md`. Copy an existing post as a template.
2. The first line is the title: `# My title`.
3. Write the body in normal Markdown.
4. Add a row for it to the experiment table in `README.md`.

## Add a figure

Save it as a PNG in `assets/img/`, then link it with a path relative to the file you're editing:

- from a post: `![what it shows](../assets/img/my_figure.png)`
- from `README.md`: `![what it shows](assets/img/my_figure.png)`

## Links

- **To another post:** `[part 2](2026-11-11-trak-noise-diagnosis.md)`, from inside `posts/`.
- **To experiment code:** `../../experiments/...`, from inside `posts/`. These links work on github.com but will **not** work on a Pages site built from `docs/`, because Pages only publishes this folder.

## Edit in the browser

On github.com, open a `.md` file, click the pencil icon, edit, then commit.

## Rules that keep it working on both github.com and Pages

- No Liquid template tags (double curly braces, or curly-brace-percent tags) and no front matter needed. Pages runs Liquid even inside code blocks, so a stray tag can break the build.
- Use relative paths only, never ones starting with `/`.
- Use PNG for figures, not interactive HTML: github.com doesn't run scripts in Markdown.

## Turning on GitHub Pages later

A repo admin goes to Settings > Pages and picks "Deploy from a branch", `main`, folder `/docs`. This hasn't been tested yet: see the note in `_config.yml`.
