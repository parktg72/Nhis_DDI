# Contract Baseline Reports

This directory holds the four approved Phase 0A/0B contract reports plus the consolidated Phase 0B baseline. Their analyzed source snapshot is `BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656`; the commit that publishes these reports is separate and may have a different SHA.

## File index

| Report | Purpose |
|---|---|
| [`phase0a-profile-contract-map.md`](./phase0a-profile-contract-map.md) | Four-profile source contract map, defaults, guards, reload behavior, and safe AST extraction. |
| [`phase0a-feature-dispersion-table.md`](./phase0a-feature-dispersion-table.md) | Exact B/F/E/D feature presence and set differences without normalizing profile boundaries. |
| [`phase0a-bundle-metadata-record.md`](./phase0a-bundle-metadata-record.md) | Source-visible bundle paths, fields, loaders, and deserialization compatibility risks. |
| [`phase0b-dependency-graph.md`](./phase0b-dependency-graph.md) | Lexical import inventory, dependency ownership, and scoped cycle analysis. |
| [`phase0b-baseline-report.md`](./phase0b-baseline-report.md) | Consolidated reproducible baseline: provenance, profile contracts, cross-profile diffs, WARNs, stale-plan corrections, and exact command/output appendices. |

## Execution order

Reproduce from the repository root with commit `BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656` available and `.venv/bin/python` reporting Python 3.12. Current `HEAD` need not equal the baseline source SHA, and the reports need not exist at that source commit. Run the appendices of [`phase0b-baseline-report.md`](./phase0b-baseline-report.md) verbatim, in this order:

1. **Appendix A** — environment, Git provenance, input SHA-256 digests, source blobs, `FEATURE_SCHEMA_LENIENT` state.
2. **Appendix B** — constant-only AST extraction; prints the deterministic JSON behind the profile contracts and B/F/E/D differences.
3. **Appendix C** — lexical import inventory of `serving/predictor.py` with scope and try-guard columns.
4. **Appendix D** — scoped static cycle scan over `serving/`, `hana_app/`, `scripts/`, `rules/` (115 modules).
5. **Appendix E** — exact Markdown allowlist, fences, local links, snapshot SHA, and analyzed source path/blob assertions; unrelated worktree paths are ignored.

Quick provenance check before running anything:

```bash
BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656
git cat-file -e "${BASELINE_SOURCE_SHA}^{commit}"
git diff --exit-code "$BASELINE_SOURCE_SHA" -- serving hana_app scripts rules
sha256sum docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md \
          docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md \
          docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md \
          docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md
```

Expected digests are recorded in section 2 of the consolidated report.

## Safety boundaries

- Commands use Git, standard-library AST parsing, source text, and Markdown validation only. Do **not** import repository modules or run artifact loaders.
- These reports do not establish deployed artifact or Windows production state. Model bytes, model-side JSON, Parquet, HANA data, `packages_win/py312/`, `mlruns/`, `out/`, and the frozen Nov→Dec holdout stay out of scope.
- `RESEARCH_TRACK_FROZEN` remains active; Gate 5A/5B and 2025-01 acquisition are retired.
- Pickle/joblib and `torch.load(..., weights_only=False)` paths execute code on load; only trusted, provenance-controlled artifacts may reach them, under a separately approved procedure.

## Known WARNs

- **Deployed artifacts uninspected** — no deployed feature order, threshold, label list, version, or hash is established here.
- **No durable test baseline** — an earlier interactive pytest run lacks durable baseline evidence. Phase 1 must capture its own node/outcome baseline.
- **Local `.venv` gaps** — `pytest==9.1.1` present, `pydantic` and `ruff` absent, so a Phase 1 baseline is not reproducible from this `.venv` as captured.

Treat these as WARNs pending separate validation, not as product contract failures. Full detail: section 8 of the consolidated report.
