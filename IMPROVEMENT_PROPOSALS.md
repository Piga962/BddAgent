# Improvement Proposals for BddAgent Pass@k Performance

## Current State Analysis

### Failure Distribution (from 64 evaluated samples)

| Category | % | Root Cause |
|----------|---|------------|
| Wrong Logic/Algorithm | 70% | Semantic misunderstanding, wrong algorithm |
| Missing Return Statement | 14% | Incomplete generation |
| Wrong API Usage | 11% | Library/framework knowledge gaps |
| Missing Edge Cases | 3% | Test case awareness |
| Missing Imports | 2% | Context limitation |

### Key Insight
Both BDD and No-BDD approaches fail identically - **the task difficulty is the limiting factor**, not the prompting strategy.

---

## Improvement Ideas

### 1. Enhanced Context Injection (Target: Wrong Logic - 70%)

**Problem**: LLM lacks sufficient context to understand exact requirements.

**Solutions**:

#### 1a. Include Ground Truth Function Signature + Docstring
```python
# Current prompt context
"Requirements: Check if value is JSON serializable"

# Enhanced context
"""
def is_json_serializable(val: Any) -> bool:
    '''Check if the input value is JSON serializable.

    JSON serializable types are: dict, list, str, int, float, bool, None

    Args:
        val: Any value to check
    Returns:
        bool: True if JSON serializable, False otherwise
    '''
    # YOUR CODE HERE
"""
```

#### 1b. Include Related Functions from Same File
```python
# Add intra_file dependencies from data.jsonl
dependency_context = get_dependencies(namespace)
prompt += f"\nRelated functions in this file:\n{dependency_context}"
```

#### 1c. Include Test Case Information
```python
# DevEval provides test paths - extract test logic
test_info = extract_test_assertions(test_path)
prompt += f"\nYour code must pass these assertions:\n{test_info}"
```

**Implementation Effort**: Medium (2-3 days)
**Expected Impact**: +15-25% Pass@1

---

### 2. Few-Shot Examples (Target: Wrong Logic - 70%)

**Problem**: LLM doesn't understand the expected code style/pattern.

**Solution**: Include 2-3 similar completed functions as examples.

```python
def build_few_shot_prompt(namespace, requirement):
    # Find similar functions from DevEval that passed
    similar_examples = find_similar_functions(namespace, k=3)

    prompt = "Here are examples of similar functions:\n\n"
    for ex in similar_examples:
        prompt += f"# {ex.namespace}\n{ex.ground_truth_code}\n\n"

    prompt += f"Now implement this function:\n{requirement}"
    return prompt
```

**Implementation Effort**: Medium (2-3 days)
**Expected Impact**: +10-20% Pass@1

---

### 3. Iterative Refinement with Execution Feedback (Target: All Categories)

**Problem**: Single-shot generation has no error correction.

**Solution**: Implement test-fix loop.

```python
def generate_with_feedback(namespace, requirement, max_iterations=3):
    code = initial_generation(requirement)

    for i in range(max_iterations):
        # Run tests
        result, error_msg = run_deveval_test(namespace, code)

        if result == "Pass":
            return code

        # Generate fix based on error
        fix_prompt = f"""
        The following code failed with error:
        {error_msg}

        Original code:
        {code}

        Fix the code to pass the test.
        """
        code = generate_fix(fix_prompt)

    return code
```

**Implementation Effort**: High (1 week)
**Expected Impact**: +20-40% Pass@1 (based on similar research)

---

### 4. Explicit Return Type Hints (Target: Missing Returns - 14%)

**Problem**: LLM forgets to return values.

**Solution**: Add explicit return requirements to prompt.

```python
# Enhanced prompt
prompt = f"""
Generate a Python function body.

Requirements: {requirements}

CRITICAL REQUIREMENTS:
1. This function MUST return a value of type: {return_type}
2. Every code path must have a return statement
3. Do not forget the final return

Example structure:
    if condition:
        return result_a
    else:
        return result_b  # <-- Don't forget this!
"""
```

**Implementation Effort**: Low (1 day)
**Expected Impact**: +5-10% Pass@1

---

### 5. Library Documentation Injection (Target: Wrong API - 11%)

**Problem**: LLM uses wrong or deprecated APIs.

**Solution**: Include relevant API documentation.

```python
def inject_api_docs(namespace, code_context):
    # Extract imports from context
    imports = extract_imports(code_context)

    # Fetch relevant documentation
    api_docs = []
    for imp in imports:
        if imp in KNOWN_LIBRARIES:
            relevant_funcs = get_relevant_functions(imp, namespace)
            api_docs.append(format_docs(relevant_funcs))

    return "\n".join(api_docs)
```

**Implementation Effort**: Medium (3-4 days)
**Expected Impact**: +5-10% Pass@1

---

### 6. Post-Generation Validation (Target: All Categories)

**Problem**: Syntactically valid but semantically broken code.

**Solution**: Multi-stage validation pipeline.

```python
def validate_and_fix(code, namespace):
    # Stage 1: Syntax validation
    if not is_valid_python(code):
        code = fix_syntax(code)

    # Stage 2: Return statement check
    if requires_return(namespace) and not has_return(code):
        code = add_return_statement(code)

    # Stage 3: Import validation
    missing_imports = check_imports(code)
    if missing_imports:
        code = add_imports(code, missing_imports)

    # Stage 4: Static analysis (pylint/mypy)
    issues = run_static_analysis(code)
    if issues:
        code = fix_issues(code, issues)

    return code
```

**Implementation Effort**: Medium (3-4 days)
**Expected Impact**: +5-15% Pass@1

---

### 7. BDD Enhancement: Concrete Test Cases (Target: Wrong Logic)

**Problem**: BDD scenarios are too abstract.

**Solution**: Generate concrete test cases, not just Given-When-Then.

```python
# Current BDD
"""
Scenario: Check JSON serializable
Given a dictionary value
When is_json_serializable is called
Then return True
"""

# Enhanced BDD with concrete cases
"""
Scenario 1: Dictionary is JSON serializable
    Input: {"key": "value"}
    Expected: True

Scenario 2: Custom object is not serializable
    Input: CustomClass()
    Expected: False

Scenario 3: List of primitives is serializable
    Input: [1, 2, "three"]
    Expected: True

Edge cases to handle:
- Nested structures
- None values
- Circular references (should return False)
"""
```

**Implementation Effort**: Low (1-2 days)
**Expected Impact**: +10-15% Pass@1

---

### 8. Model Ensemble / Self-Consistency

**Problem**: Single generation may have errors.

**Solution**: Generate multiple candidates and select best.

```python
def ensemble_generation(requirement, n_samples=5):
    candidates = []
    for _ in range(n_samples):
        code = generate_code(requirement, temperature=0.7)
        candidates.append(code)

    # Score candidates
    scores = []
    for code in candidates:
        score = 0
        if is_valid_python(code): score += 1
        if has_return_statement(code): score += 1
        if passes_static_analysis(code): score += 1
        scores.append(score)

    # Return best candidate
    return candidates[scores.index(max(scores))]
```

**Implementation Effort**: Low (1 day)
**Expected Impact**: +5-10% Pass@1

---

## Priority Ranking

| Rank | Improvement | Effort | Expected Impact | ROI |
|------|-------------|--------|-----------------|-----|
| 1 | Iterative Refinement | High | +20-40% | High |
| 2 | Enhanced Context | Medium | +15-25% | High |
| 3 | Concrete BDD Tests | Low | +10-15% | Very High |
| 4 | Few-Shot Examples | Medium | +10-20% | Medium |
| 5 | Explicit Returns | Low | +5-10% | Very High |
| 6 | Post-Generation Validation | Medium | +5-15% | Medium |
| 7 | API Docs Injection | Medium | +5-10% | Low |
| 8 | Model Ensemble | Low | +5-10% | High |

---

## Quick Wins (Implementable Today)

### 1. Add Return Type Requirements
Edit `run_ablation.py` to include return type in prompt.

### 2. Add Concrete Examples to BDD
Modify BDD prompt to generate concrete input/output pairs.

### 3. Multiple Generation + Best Selection
Generate 3 candidates, pick the one that passes syntax validation.

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Explicit return requirements in prompts
- [ ] Concrete BDD test case generation
- [ ] Basic syntax validation before saving

### Phase 2: Context Enhancement (3-5 days)
- [ ] Include function signature and docstring
- [ ] Add intra-file dependencies
- [ ] Extract and include test assertions

### Phase 3: Iterative Refinement (1 week)
- [ ] Implement test execution feedback loop
- [ ] Error-guided code repair
- [ ] Multi-iteration generation

### Phase 4: Advanced (2 weeks)
- [ ] Few-shot example retrieval system
- [ ] API documentation injection
- [ ] Model ensemble with voting

---

## Expected Results After Improvements

| Phase | Estimated Pass@1 | Improvement |
|-------|------------------|-------------|
| Current | 0% | Baseline |
| Phase 1 | 5-10% | +5-10% |
| Phase 2 | 15-25% | +10-15% |
| Phase 3 | 30-40% | +15-20% |
| Phase 4 | 40-50% | +10-15% |

Note: These are estimates based on similar research. Actual results may vary.
