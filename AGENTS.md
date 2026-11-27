# Core Principles

Correctness: Tests exist and pass, and the behavior matches the stated requirement including edge cases.
Security: No secrets in code; inputs are validated; any tools or external calls use least-privilege access.
Maintainability: The code reads clearly, follows team conventions, and contains no unexplained complexity.
Human understanding: The developer submitting the change can explain what the code does and why, including how it handles the inputs it was not explicitly tested against. 
Responsible AI: AI is not allowed to stage / commit / push code. Only human is allowed to.
Transparency: No code is written outside this repository, disposable temporary code/files must be written to `./tmp`
Atomic & Idempotent: Commit only happens when the code succeeds, otherwise rollback. Ensure that a re-run always return the same output
No fallbacks: Minimize creating unnecessary fallbacks (defensive scaffholding) unless the user asked to
DUMB: Follow the D.U.M.B. (Descriptive, Uniform, Minimal, and Basic.) principle
KISS: Code must be kept simple and stupid
So what: Use declarative plot title, section titles that adds interpretive power for the reader. Readers should understand the key takeaways with ease.

## Research workflow

This pipeline supports research. It is not a production misinformation detector. The hillclimbing series has ended. Do not continue the series or schedule milestone reviews. Keep `hillclimbing.md` as a record of past experiments, not a work queue.

Use fact-check retrieval to find candidates for human review. Do not treat similarity scores or match counts as evidence of hoax prevalence. If the study needs a prevalence estimate, label a representative sample that includes posts the matcher did not flag. Run further evaluations only when they answer a specific research question.