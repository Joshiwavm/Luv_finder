---
name: notion-sync
description: Bring the Luv-finder Notion pages (the method chapter of the A&A paper draft and its JOSS-article subpage) up to date with main. Use it after a pull request merges into main, when a merge event for a Luv_finder PR arrives, when the user says the Notion or the paper draft is stale, asks to update, sync or refresh Notion, the write-up, the methods draft or the JOSS draft, or runs /notion-sync.
---

# Notion sync

The Luv-finder Notion page is the method chapter of the methods paper (A&A); its subpage
is the JOSS software paper. Both quote code, equations and numbers from `main`, so every
merge can leave an excerpt, a formula or a figure wrong. This skill updates only what
the merged changes touched, so the user's own edits on the pages survive.

- Main page: https://app.notion.com/p/3e546a617d0081c78b93d88adcc98ce2
- JOSS subpage: https://app.notion.com/p/3f746a617d0081c8a04ec33593015f57

## Steps

1. **Find the last sync.** Fetch the main page. Its last line reads
   ``Synced with main @ `<sha>` (<date>)``. Without that line, stop and ask the user before
   rewriting anything.
2. **See what changed.**
   ```bash
   git fetch origin main
   git log --oneline <sha>..origin/main
   git diff --stat <sha>..origin/main
   ```
   No new commits: report "nothing to sync" and stop. Otherwise read the diffs of the files
   in the map below and the commit messages; the messages often record the measured
   numbers (timings, thresholds, line counts) that the results sections quote.
3. **Map files to sections.**

   | Changed | Section on the main page |
   |---|---|
   | `luv_finder/data.py` | 1 Data; 5 (`channel_correlation`); 6 Null hypothesis (`jackknife`) |
   | `luv_finder/model.py` | 2 Detection problem |
   | `luv_finder/kernel.py` | 3 Spatial collapse; 9 Line fit (`_moments`) |
   | `luv_finder/matchedfilter.py` | 4 Line template and continuum; 5 Correlated channel noise; 7 Primary beam and mosaics |
   | `luv_finder/utils.py` | 7 (primary beam) |
   | `luv_finder/catalogue.py` | 8 From peaks to a catalogue |
   | `luv_finder/fit.py` | 9 Line fit |
   | `tests/` | 10 Validation |
   | `ROADMAP.md`, result-bearing commit messages | 10-12 (ASPECS, SPT-CL J0459-4947, open items) |
   | new module, CLI option, dependency or performance figure | intro of the main page; JOSS subpage |

   A refactor that leaves behaviour, names and the quoted excerpts unchanged needs no
   edit; say so in the report rather than touching the page.
4. **Edit the main page** with `notion-update-page`, command `update_content`, one
   `old_str`/`new_str` pair per changed paragraph, equation or code block. Copy `old_str`
   from the fetched text. Keep the section order and headings. If a passage differs from
   what the previous code would have produced, the user edited it: merge the code change
   into their wording instead of overwriting it.
   - Code excerpts are copied verbatim from `git show origin/main:<file>` (trimmed, never
     paraphrased), with a first comment naming the file and function.
   - Numbers come from `ROADMAP.md`, the tests or commit messages, and results carry the
     date of their run.
5. **Edit the JOSS subpage** only when features, modules, dependencies, performance
   figures or research impact change. Keep the structure of the official example paper
   (front matter, Summary, Statement of need, State of the field, Software design,
   Research impact statement, AI usage disclosure, Acknowledgements, References), 750-1750
   words, and no derivations: it points to the main page for the math.
6. **Verify.** Re-fetch both pages. Every changed equation renders (no literal `$``$`, no
   `\{` escapes, no list collapsed into `<br>`), every excerpt matches `origin/main`, and the
   `<page>` link to the JOSS subpage is still there.
7. **Bump the sync line** to `git log -1 --format='%h %ad' --date=short origin/main`.
8. **Report** one line per section changed, and what was left alone and why.

If the Notion tools are unavailable, say so and list the edits you would have made.

## Writing rules

From `CLAUDE.md`: the pages read like a JOSS paper (explanation, math and code
interleaved), in the terminology of Vio & Andreani (2021, arXiv:2107.09378): H0/H1,
template s, statistic T(x), SNR, P_FA, threshold gamma. Concise and accurate; state each
thing once, also across the two pages.

## Notion markdown pitfalls

- Inline math is `` $`...`$ ``. A missing closing backtick garbles everything up to the next
  block: the following list collapses into `<br>` and display math gets escaped. Before
  sending, check that every line has as many `` $` `` as `` `$ ``.
- Display math is `$$`, the equation, `$$`, each on its own line.
- Escape `[ ] < > { } | ^ * ~ $` in plain text (citations: `\[@key\]`), not inside code or
  math.
- A `---` divider next to a `<page>` block can move; keep dividers away from page links.
- `replace_content` deletes child pages whose `<page url=...>` tag is missing from the new
  content, so prefer `update_content`. If a garbled block cannot be matched, replace the
  whole page with the JOSS `<page>` tag kept and `allow_deleting_content` left false, so the
  call fails instead of deleting the subpage.
