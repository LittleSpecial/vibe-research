Convert the idea into an executable plan.
Return STRICT JSON array where each element has:
- id (string)
- title (string)
- goal (string)
- commands (array of shell commands)
- expected_output (string)
- est_gpu_hours (number, TOTAL gpu-hours for that item)
- priority ("high"|"medium"|"low")

Budget constraints:
- Hardware limit: up to configured A100 GPUs.
- Total estimated GPU-hours across all items must fit (configured_gpu_limit * wall_clock_budget_hours).
