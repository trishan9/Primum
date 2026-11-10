# Primum web: design direction

Direction given by the project brief. This file records how it was read and the reason for each
major decision, so later changes can be judged against something.

**Design read.** A research instrument (three working screens and one long-form method page) for
academic reviewers and clinically literate readers, in a typeset-journal-meets-lab-instrument
language. Dials: energy 1 (calm), rhythm 2 (each view has its own composition), motion 1 (feedback only).

## The idea

The unusual thing about this product is that it measures its own confidence against a certified
threshold and sometimes declines to answer. So the interface is built around one reading:
**a pointer on a ruled scale, against a gate.** Left of the gate the scale is hatched, and an
answer whose pointer lands there is withheld. The same hatch marks every withheld thing (the
band on the abstention panel, the withheld region of the explainer chart), so abstention is
legible without colour. The product mark is that pointer and scale.

## Decisions and their reasons

| Decision | Reason |
| --- | --- |
| Achromatic chrome; near-black primary action | The data is the only thing allowed to be loud. A reviewer should see evidence and verdicts, not a brand. |
| One data hue, blue (`--accent`) | It marks the confidence pointer and the lead series of a chart. Blue stays distinguishable from the three verdict colours under colour-vision deficiency. |
| Green, amber, red only for claim verdicts and the emergency state | Status colour is reserved for status, always paired with a glyph and a word. Abstention is not an error, so it is never red. |
| STIX Two Text for anything the system says in prose (answers, claims, passages, the method) | It is the typeface family built for scientific publishing, and it carries Greek and mathematics properly: α, δ and λ̂ are content here. An answer should read like a paragraph of the literature it cites. |
| IBM Plex Sans for controls, labels and every figure | Drawn for technical documentation, with true tabular figures, and clearly distinct from the serif. |
| IBM Plex Mono for shell commands and identifiers only | Commands and PMIDs are code; nothing else is. |
| No sidebar: a top bar with four destinations | There are four views. A sidebar would spend 240px on four links. |
| Ask is three columns: pipeline, reading, evidence | They are the three questions a sceptical reader asks, in order: what did it do, what did it conclude, on what basis. |
| Results tables are forest plots | A point estimate with its interval, one row per system, is how the clinical literature compares effects. |
| The retrieval lab draws lines between rankings | Rank changes are the subject of that view, so they are drawn, not left to be inferred from two lists. |
| Method page sets analogies as margin notes on wide screens | The way a journal sets a side note: beside the paragraph it glosses, out of the argument's way. |
| 8px cards, 6px controls, 4px chips; 1px hairlines; no shadows at rest | One tight radius family. Depth comes from a single tone step. Only tooltips float, so only they cast a shadow. |
| Dashes mean "reference" (threshold, target, baseline); grids are solid hairlines | A dashed line always carries the same meaning. |
| Motion: the pointer settles on the scale, a caret while text streams, a ring on the running stage | Each shows a change of state. Nothing performs an entrance. All of it stops under `prefers-reduced-motion`. |
| Light and dark themes, following the system until the reader chooses | No brand reason fixes a theme; reviewers read in both. Dark is true neutral, not navy. |

## Integrity rule in the interface

No number is typed into this codebase as a result. Experiments and the threshold explainer read
`/api/experiments/*` and `/api/calibration`; when those answer 404 or do not answer, the page says
which command produces the file and draws nothing. The only invented content is the development
fixture for the Ask view (`NEXT_PUBLIC_FIXTURES=1`), which is off by default, bannered on screen,
and loaded as a separate chunk only when the flag is set.
