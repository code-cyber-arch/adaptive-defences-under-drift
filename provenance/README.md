# Origin and migration boundary

The original benchmark was adopted from [code-cyber-arch/adversarial-drift-detection](https://github.com/code-cyber-arch/adversarial-drift-detection), commit `c611fa1860f03bd8c4e72c4ac7754c276b430c85`, under `drift-response-2/`. The local continuation added the completed filter study, matched no-reset controls and their separate audits.

This standalone repository was prepared on 12 September 2026. It starts a new Git history under the name **adaptive-defences-under-drift**. It does not rewrite the original history or modify the original folder. The current thesis numbering is retained.

- `retained-files.json` records the public files copied from the prior project and their hashes. The original experimental code and retained measurements remain unchanged.
- `local-artifacts.json` records retained large data, model and trace files excluded from Git.
- The author's migration checks and exact previous-to-new file inventory are retained locally under `.local/migration/`.

Excluded from the standalone copy: the stopped partial benchmark backup, old smoke outputs, compiler binaries and caches, temporary files, duplicate per-run presentation exports, superseded reports and editorial review backups. No retained research trace was removed to achieve the compact Git repository. Existing archive gaps are disclosed in [the limitations](../docs/LIMITATIONS.md).

The current thesis source, PDF, references and figures were copied to `.local/thesis/` for the author to continue writing. That directory is excluded from Git. Examiner documents, marking sheets and thesis exemplars are not public repository content.
