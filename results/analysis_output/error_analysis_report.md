# LiveCodeBench Error Analysis Report

## gpt-4o

- **Problems**: 50
- **Pass Rates**: BDD=16.0%, CoT=2.0%, Direct=16.0%

### Error Distribution (CoT)

- NO_OUTPUT: 37 (74.0%)
- WRONG_ANSWER: 8 (16.0%)
- OTHER_ERROR: 4 (8.0%)
- PASSED: 1 (2.0%)

### CoT Catastrophe Examples

Problems where CoT failed but Direct succeeded:

**A. Short Sort** (easy)
- CoT Error: NO_OUTPUT
- CoT Tokens: 1654, Direct Tokens: 660

**B. Good Kid** (easy)
- CoT Error: WRONG_ANSWER
- CoT Tokens: 1570, Direct Tokens: 575

**sum-in-a-matrix** (medium)
- CoT Error: NO_OUTPUT
- CoT Tokens: 1479, Direct Tokens: 560


---

## gpt-5.3-codex

- **Problems**: 50
- **Pass Rates**: BDD=26.0%, CoT=4.0%, Direct=54.0%

### Error Distribution (CoT)

- OTHER_ERROR: 48 (96.0%)
- PASSED: 2 (4.0%)

### CoT Catastrophe Examples

Problems where CoT failed but Direct succeeded:

**B. Good Kid** (easy)
- CoT Error: OTHER_ERROR
- CoT Tokens: 585, Direct Tokens: 496

**D. 1D Eraser** (easy)
- CoT Error: OTHER_ERROR
- CoT Tokens: 777, Direct Tokens: 700

**B. Chemistry** (medium)
- CoT Error: OTHER_ERROR
- CoT Tokens: 1373, Direct Tokens: 1145


---

## gpt-35-turbo

- **Problems**: 50
- **Pass Rates**: BDD=30.0%, CoT=24.0%, Direct=22.0%

### Error Distribution (CoT)

- OTHER_ERROR: 25 (50.0%)
- PASSED: 12 (24.0%)
- WRONG_ANSWER: 11 (22.0%)
- NO_OUTPUT: 2 (4.0%)

### CoT Catastrophe Examples

Problems where CoT failed but Direct succeeded:

**D. Yarik and Musical Notes** (hard)
- CoT Error: WRONG_ANSWER
- CoT Tokens: 1604, Direct Tokens: 1547

**find-the-losers-of-the-circular-game** (easy)
- CoT Error: OTHER_ERROR
- CoT Tokens: 1265, Direct Tokens: 793


---

## gpt-4.1

- **Problems**: 50
- **Pass Rates**: BDD=38.0%, CoT=16.0%, Direct=16.0%

### Error Distribution (CoT)

- NO_OUTPUT: 33 (66.0%)
- PASSED: 8 (16.0%)
- WRONG_ANSWER: 7 (14.0%)
- OTHER_ERROR: 2 (4.0%)

### CoT Catastrophe Examples

Problems where CoT failed but Direct succeeded:

**number-of-senior-citizens** (easy)
- CoT Error: WRONG_ANSWER
- CoT Tokens: 897, Direct Tokens: 511

**sum-in-a-matrix** (medium)
- CoT Error: NO_OUTPUT
- CoT Tokens: 982, Direct Tokens: 516

**find-the-punishment-number-of-an-integer** (medium)
- CoT Error: NO_OUTPUT
- CoT Tokens: 738, Direct Tokens: 628


---

## gemini-3.1-flash-lite-preview

- **Problems**: 50
- **Pass Rates**: BDD=66.0%, CoT=60.0%, Direct=30.0%

### Error Distribution (CoT)

- PASSED: 30 (60.0%)
- NO_OUTPUT: 10 (20.0%)
- WRONG_ANSWER: 6 (12.0%)
- OTHER_ERROR: 4 (8.0%)

### CoT Catastrophe Examples

Problems where CoT failed but Direct succeeded:

**construct-the-longest-new-string** (medium)
- CoT Error: WRONG_ANSWER
- CoT Tokens: 1135, Direct Tokens: 1004


---

