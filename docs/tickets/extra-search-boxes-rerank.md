# Extra — The right object boxed on each search result

**Not a slice · Mon 23 Nov – Tue 1 Dec 2026 · branch `search-boxes-rerank`**
Board: OmniSight Phase B · Assignee: Cyrine

> **By Tue 1 Dec:** search *"a blue car"* and the box is drawn around **the blue car**, even
> when it is small and far away, not around the nearest car. The results appear first and the
> box follows quickly. This is what the supervisor sees on Wed 2 Dec.

**Builds on:** OMNI-X02 (`docs/tickets/extra-search-boxes.md`, decision D-42).
**Ships in:** `services/ai_workers/` · `services/query/` (the boxes route only) · `frontend/`
(the box drawing only)
**Not in this ticket:** a box that follows the object while the video plays (tracking);
changing the search itself.

---

## The plan, day by day

| Day | Ticket |
|---|---|
| **Mon 23 Nov** | OMNI-X06 · Branch, setup and baseline |
| **Tue 24 Nov** | OMNI-X07 · Measure where the time goes |
| **Wed 25 Nov** | OMNI-X08 · YOLO-World proposes more candidates |
| **Thu 26 Nov** | OMNI-X09 · SigLIP 2 scores each candidate |
| **Fri 27 Nov** | OMNI-X10 · One box per result, and D-48 |
| **Sat 28 Nov** | Catch up on anything left from the week |
| **Sun 29 Nov** | OMNI-X11 · Speed |
| **Mon 30 Nov** | OMNI-X12 · The check, before and after |
| **Tue 1 Dec** | OMNI-X13 · Docs and the pull request |

## Why this ticket

The search finds the right frame: SigLIP 2 understands *blue*. But the box is often on the
wrong object. YOLO-World finds *cars* well and ignores colours and details, so it boxes the
biggest, clearest cars (the near ones), and we keep its top 3 by its own score. The boxes also
appear late: they are a second request after the results (`GET /search/boxes`), run on the
laptop's CPU, and get slower still when `ai_workers` is busy encoding frames (K-25).

## Where the code is today

| Part | File |
|---|---|
| The detector (YOLO-World) | `services/ai_workers/app/detector.py` (`BoxDetector.detect`) |
| The detect route | `services/ai_workers/app/main.py` (`POST /detect`) |
| SigLIP 2 | `services/ai_workers/app/encoder.py` (`FrameTextEncoder`: `encode_images`, `encode_texts`) |
| The boxes route | `services/query/app/search.py` (`GET /search/boxes`) and `services/query/app/ai_workers_client.py` (`detect_boxes`; the text-vector cache) |
| The drawing | `frontend/src/components/box-overlay.tsx`, `moment-search.tsx` |

## Working beside Ahmad

Ahmad works on the forecasting model and the Alerts page at the same time
(`docs/tickets/week-09-10.md`, branch `slice-09-10-forecast`).

- This ticket touches only the files above.
- Do not edit `services/pfm/`, the alerts routes and tables, or the Alerts page.
- **This ticket's decision is D-48.** (Ahmad's sprint starts at D-49; D-46 is reserved.)
- Shared files (`docs/results.md`, `docs/decisions.md`, `.env.example`): small additions only.

---

## Rules for this ticket

- **You run every git command.** Claude Code suggests a short commit message after each
  ticket, without the ticket number.
- **Only `ai_workers` touches a model.** `query` calls it.
- **The results never wait for the boxes.** Boxes are a separate request that fills in after.
- **Measure before changing**, and report the real numbers, even slow ones.

---

## OMNI-X06 · Branch, setup and baseline · Mon 23 Nov

**Why:** start from a working system and a known state, so every later change can be
compared.

- [ ] Pull the latest `master` and create the branch `search-boxes-rerank` (you run it)
- [ ] The stack runs: `docker compose up -d`, open `https://localhost`, sign in
- [ ] A search on a VIRAT video shows boxes on its results (the X02 behaviour)
- [ ] `uv run pytest` passes; note the count
- [ ] Read OMNI-X02's section in `docs/context/` and D-42 in `docs/decisions.md`

**Done when:** the stack runs, a search shows boxes, and the tests pass.

## OMNI-X07 · Measure where the time goes · Tue 24 Nov

**Why:** the boxes appear late, and we must know which step is slow before changing anything.

- [ ] One search with 5 results on a VIRAT video. Time each step: the whole `GET
      /search/boxes` request, YOLO-World per frame, and the time spent waiting for
      `ai_workers`
- [ ] Twice: with `ai_workers` **idle**, and **busy** (encoding frames of a new upload)
- [ ] 3 runs each; report the middle and the slowest
- [ ] Written in `results.md` as a short entry: machine, settings, numbers, what was not
      covered

**Done when:** the times are in `results.md`.

## OMNI-X08 · YOLO-World proposes more candidates · Wed 25 Nov

**Why:** today we keep only YOLO-World's top 3 by its own score, so a small, far object
(the blue car) never makes the list.

- [ ] For the object word of the query (the X02 list: car, person, bag...), keep **all** its
      boxes above a **lower threshold** (a new setting, starting lower than today's 0.25)
- [ ] A **cap** on candidates per frame (a setting, e.g. 10), so one crowded frame cannot slow
      everything down
- [ ] These are **candidates**, not shown on screen yet; the screen does not change today
- [ ] Tests: on a stored VIRAT frame with a far car, the far car is among the candidates

**Done when:** small, far objects appear among the candidates.

## OMNI-X09 · SigLIP 2 scores each candidate · Thu 26 Nov

**Why:** SigLIP 2 understands colours and details ("blue"), so it can tell which candidate
matches the words.

- [ ] Crop each candidate box from the stored frame, with a small margin (a setting)
- [ ] Encode the crops with SigLIP 2's image side, **in one batch** (add a method that takes
      images in memory, beside `encode_images`)
- [ ] Score each crop against **the query's SigLIP 2 text vector**: the one the search
      already made (`query` caches it in `ai_workers_client.py`); pass it to `/detect`, do not
      encode the text again
- [ ] Each candidate now has a match score (cosine)
- [ ] Tests: the crop's position, the batch, the score

**Done when:** each candidate has a match score.

## OMNI-X10 · One box per result, and D-48 · Fri 27 Nov

**Why:** one right box is clearer than three boxes, two of them wrong.

- [ ] Each result shows **one box**: the candidate with the best match score
- [ ] **A floor:** if even the best score is below it (a setting), show no box, so a wrong
      box is never forced
- [ ] The label stays the object word; the match score on hover
- [ ] The response of `GET /search/boxes` keeps its shape (a list per result), so the screen
      changes as little as possible
- [ ] Record **D-48** in `decisions.md`: why SigLIP 2 re-ranks YOLO-World's boxes, the
      settings (lower threshold, cap, margin, floor) and their starting values

**Done when:** *a blue car* boxes the blue car on screen.

## Sat 28 Nov · Catch up

Finish anything left from X06–X10. Nothing new.

## OMNI-X11 · Speed · Sun 29 Nov

**Why:** the fix adds work (crops and SigLIP 2), so it must not make the boxes slower than
they need to be.

- [ ] The crops of all 5 results encoded **together** in one batch, not one result at a time
- [ ] The results show first and the boxes fill in after, as today; a slow box never delays
      the results
- [ ] Measure again exactly as in X07: idle and busy, 3 runs each
- [ ] Before and after side by side in `results.md`. On a GPU it will be much faster; on the
      laptop, report the number as it is

**Done when:** the new times are in `results.md`.

## OMNI-X12 · The check, before and after · Mon 30 Nov

**Why:** the point is the right box, so it is checked by eye on real searches.

- [ ] 4 searches on the VIRAT videos: *a blue car*, *a white truck*, *a person carrying a
      bag*, *a person walking*
- [ ] For each top 5: box on the right object, wrong, or missing, **before** (X02,
      `master`) and **after** (this branch). A small table in `results.md`
- [ ] At least one case where the right object is small and far away
- [ ] Screenshots kept (before and after)
- [ ] `uv run pytest` passes, with the new tests

**Done when:** every check above is ticked.

## OMNI-X13 · Docs and the pull request · Tue 1 Dec

- [ ] `results.md`: the entries complete (X07, X11, X12)
- [ ] `decisions.md`: D-48 complete
- [ ] `docs/context/`: one section for this ticket (why, what was done, what was verified,
      what was not, result)
- [ ] Push the branch and open a **pull request** to `master` for Ahmad (you run it)

**Done when:** Ahmad has the pull request before the meeting.

---

## Working with Claude Code

`CLAUDE.md` is read automatically. One ticket per session. Paste the ticket, then:

> Here is ticket OMNI-X0x from our board. Implement exactly what it asks and nothing more.
> Before starting, read the context and decisions for OMNI-X02 (D-42). When you are done,
> show me the diff, tell me which checks you verified and how, and suggest a short commit
> message without the ticket number. Do not run any git command that changes the repository;
> I commit myself.
