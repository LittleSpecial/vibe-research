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
- Hardware limit: up to 4xA100.
- Total estimated GPU-hours across all items must fit 4 * wall-clock_budget_hours.
