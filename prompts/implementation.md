Produce a runnable implementation package.
Output two sections:
1) Markdown checklist
2) Shell script fenced block (bash) that can run on Linux aarch64 with Slurm.

Requirements:
- The script must be practical and executable with minimal manual edits.
- Respect hardware limit: up to configured A100 GPUs.
- Save core metrics to JSON under runs/<RUN_ID>/results.json.
- Include at least one strong baseline and one ablation in the run plan.
