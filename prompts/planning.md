Convert the idea into an executable plan.
Return STRICT JSON array where each element has:
- id (string)
- title (string)
- goal (string)
- commands (array of shell commands)
- expected_output (string)
- est_gpu_hours (number)
- priority ("high"|"medium"|"low")

Keep total estimated GPU hours <= 12.
