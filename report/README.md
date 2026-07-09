# Technical report

- **`report.md`** — the 15-page technical report (source; Pandoc-flavored Markdown with
  YAML front-matter, LaTeX math, tables, and figure references).
- **`report.html`** — self-contained render (figures embedded); open in a browser to read or
  **Print → Save as PDF**.
- **`figures/`** — all figures referenced by the report.

## Build the PDF

```bash
make report            # from repo root
```

`make report` uses Pandoc. With a LaTeX engine installed (`texlive`/`tectonic`) it produces
`report/report.pdf`; otherwise it falls back to `report.html`. On Colab (which ships TeX):

```bash
!apt-get -qq install -y texlive-xetex >/dev/null && \
  pandoc report/report.md -o report/report.pdf --toc -V geometry:margin=1in
```

Or simply open `report.html` and use the browser's *Print → Save as PDF*.
