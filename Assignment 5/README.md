# Assignment 5 Submission

## Files included
- `main.py` — complete solution for Tasks 1, 2, 3, and 4
- `requirements.txt` — dependencies
- `run_instructions.txt` — quick run steps
- `analysis_results.txt` — generated after running the code
- `task1_policy_values.png` — bar chart for Task 1.3
- `task2_value_iteration_convergence.png` — convergence plot for Task 2.3
- `task3_policy_iteration_values.png` — line plot for Task 3.3(a)
- `task3_policy_iteration_policy.png` — policy heatmap/table for Task 3.3(b)

## Important note
The assignment asks for `R` to be a 2D array of shape `(3, 2)`, but the transition table gives different rewards for `Low -> Search` depending on the next state:
- to `High`: `-3`
- to `Low`: `+4`

Because the task explicitly requires a 2D `R[s, a]`, the code uses the expected immediate reward:

`R[Low, Search] = 0.4*(-3) + 0.6*(4) = 1.2`

This keeps the implementation exactly in the format requested.

## How to run
```bash
pip install -r requirements.txt
python main.py
```
