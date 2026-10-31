# Paper draft

ACL template: `acl.sty` and `acl_natbib.bst` from github.com/acl-org/acl-style-files @ d5adc82 (2026-06-29), unchanged.

Build (tectonic fetches packages and runs BibTeX itself):

    cd paper && tectonic -o build main.tex      # -> build/main.pdf

Rules:
- Every number is a macro from `numbers.tex` (`\val{Name}`, `\ci{Name}` fail the build if the macro is missing)
  or a generated table in `tables/`. Never type a result. `legacy_numbers.tex` is never `\input`.
- Regenerate numbers/tables/figures with `python scripts/make_tables.py && python scripts/make_findings.py`.
- `references.bib` is copied unchanged from `docs/related_work.md`; add citations there first (verified).
- Sections waiting on results are `\todo{}` stubs (results, ICT, setup, conclusion).
