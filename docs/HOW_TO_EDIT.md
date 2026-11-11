# How to edit the experiment log

This folder is a Jekyll blog. GitHub Pages builds it automatically, so there is no build step and nothing to install. Everything is plain Markdown.

## Where things are

| file | what it is |
|---|---|
| `index.md` | the dashboard (front page): the experiment table, headline numbers, figures |
| `_posts/YYYY-MM-DD-title.md` | one post per experiment or write-up |
| `assets/img/` | figures used in posts |
| `_config.yml` | site title, description, `baseurl` |

## Add a new post

1. Copy an existing file in `_posts/` and rename it `YYYY-MM-DD-short-title.md`. The date in the file name sets the post date and its URL.
2. Edit the header between the `---` lines (title, date, tags).
3. Write the body in Markdown. Everything above `<!--more-->` is the excerpt shown on the front page.
4. Add a row for the post to the experiment table in `index.md`.

## Add a figure

Put the PNG in `assets/img/`, then reference it like this:

```markdown
![alt text]({{ '/assets/img/my_figure.png' | relative_url }})
```

Use the `{{ ... | relative_url }}` form, not a bare `/assets/...` path. It adds the site's `baseurl`, so the link works on GitHub Pages.

## Link to another post

```markdown
[part 2]({{ '/2026/11/11/trak-noise-diagnosis/' | relative_url }})
```

The URL pattern is `/year/month/day/<file name without date and .md>/`.

## Edit in the browser

On GitHub, open any `.md` file and click the pencil icon, then commit. The site rebuilds in about a minute.

## Preview locally (optional)

This needs Ruby, which this machine doesn't have:

```bash
gem install bundler jekyll github-pages
jekyll serve --source docs --baseurl ""
```

## Deploying somewhere other than royk999.github.io/Retrace

Change `baseurl` in `_config.yml` to `"/<repo-name>"`, or to `""` for a `<user>.github.io` repo.
