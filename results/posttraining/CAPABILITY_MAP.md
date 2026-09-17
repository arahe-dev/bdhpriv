# Arm-A post-trained model: capability map

Battery frozen before evaluation (SHA-256 589866c68ad76ec017fe5d41bb2b67b86541be24899599fcb4d837e8d490beb3), 162 items, 18 families, programmatic grading; no checkpoint selection used this battery. Checkpoint: pt_006_dose010 (lr 3e-4, 50/50 tasks+TinyStories, 0.10 TPP). Base = frozen 2.5B.

| family | base | pt_006_dose010 | interpretation |
|---|---|---|---|
| arith_ood_add | 0.00 | 0.00 | absent |
| arith_ood_div | 0.00 | 0.00 | absent |
| arith_ood_mul | 0.00 | 0.00 | absent |
| arith_ood_sub | 0.00 | 0.00 | absent |
| arith_train | 0.00 | 0.25 | partial, unreliable |
| completion | 0.00 | 0.00 | lost (task-format dominance) |
| context_extract | 0.00 | 0.20 | weak |
| copy | 0.00 | 1.00 | works |
| count | 0.00 | 1.00 | works |
| facts | 0.00 | 0.00 | absent |
| first_letter | 0.00 | 1.00 | works |
| format | 0.00 | 0.33 | weak |
| listing | 0.00 | 0.12 | absent (maps to count) |
| nonsense | 0.60 | 1.00 | robust |
| reverse | 0.00 | 1.00 | works |
| sort | 0.00 | 0.80 | works |
| story | 0.00 | 1.00 | works (fluent, templated) |
| transfer | 0.00 | 0.50 | half transfers to new wording |
| **macro** | 0.033 | 0.456 | |

## Carry breakdown (2-digit addition)
- with carry: 1/4
- without carry: 4/16

## Example outputs (greedy)

**copy**
- Q: Copy this text: rock read cold -> rock read cold (expected rock read cold)
- Q: Copy this text: tree leaf paint prince tree -> tree leaf paint prince tree (expected tree leaf paint prince tree)
- Q: Copy this text: soup sand bean queen -> soup sand bean queen (expected soup sand bean queen)
- base: Copy this text: rock read cold -> Myths: Manuals: Manuals

**arith_train**
- Q: Calculate: 74 + 31 = -> 105 (expected 105)
- Q: Calculate: 48 + 86 = -> 130 (expected 134)
- Q: Calculate: 17 + 36 = -> 51 (expected 53)
- base: Calculate: 74 + 31 = -> Joe Biden

**context_extract**
- Q: The secret word is happy. Question: What is the secret word? -> hgual (expected happy)
- Q: The secret word is bitter. Question: What is the secret word? -> rats (expected bitter)
- Q: The secret word is moon. Question: What is the secret word? -> moon (expected moon)
- base: The secret word is happy. Question: What is the se -> Joe Barry

**completion**
- Q: The cat sat on the -> Did you see a big, (expected ['mat', 'floor', 'couch', 'chair', 'bed', 'ground', 'table', 'lap', 'roof', 'windowsill'])
- Q: The sun rises in the -> Once upon a time, there was (expected ['east', 'morning', 'sky'])
- Q: Water freezes when it gets -> looking (expected ['cold', 'freezing', 'colder'])
- base: The cat sat on the -> The cat sat on the

**facts**
- Q: What is the capital of France? -> France (expected ['paris'])
- Q: What is the capital of India? -> nasirb (expected ['new delhi', 'delhi'])
- Q: What is the capital of the United States? -> lawy (expected ['washington', 'washington dc', 'washington d.c.'])
- base: What is the capital of France? -> France

**listing**
- Q: List three colors. -> 2 (expected colors)
- Q: Name two colors. -> 2 (expected colors)
- Q: Give me a color. -> Once upon a time, there was a little girl named Lily. She loved to pla (expected colors)
- base: List three colors. -> â€œâ€

## Conclusions
- Solid narrow skills: exact copy, word count, first letter, reversal, sorting (80%), fluent micro-stories (100%).
- Arithmetic is only partially learned (25% at 2 digits) and does not generalize beyond trained ranges/operations.
- No factual or category knowledge; category requests collapse into the count task.
- Post-training narrowed the distribution: cloze completion now fails through task-format interference.
- Instruction-following beyond the trained task templates is weak (format compliance 33%, context extraction 20%).