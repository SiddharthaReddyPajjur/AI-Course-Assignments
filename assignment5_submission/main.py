import numpy as np
import matplotlib.pyplot as plt

STATES = ['High', 'Low', 'Charging']
ACTIONS = ['Search', 'Wait']
S, A = len(STATES), len(ACTIONS)

GAMMA = 0.9
THETA = 1e-6

state_to_idx = {s: i for i, s in enumerate(STATES)}
action_to_idx = {a: i for i, a in enumerate(ACTIONS)}

valid_actions = {
    state_to_idx['High']: [action_to_idx['Search'], action_to_idx['Wait']],
    state_to_idx['Low']: [action_to_idx['Search'], action_to_idx['Wait']],
    state_to_idx['Charging']: [action_to_idx['Wait']],
}

def build_mdp():
    P = np.zeros((S, A, S), dtype=float)
    R = np.zeros((S, A), dtype=float)

    h, l, c = state_to_idx['High'], state_to_idx['Low'], state_to_idx['Charging']
    search, wait = action_to_idx['Search'], action_to_idx['Wait']

    P[h, search, h] = 0.7
    P[h, search, l] = 0.3
    R[h, search] = 4

    P[h, wait, h] = 1.0
    R[h, wait] = 1

    P[l, search, h] = 0.4
    P[l, search, l] = 0.6
    R[l, search] = 0.4 * (-3) + 0.6 * 4

    P[l, wait, l] = 1.0
    R[l, wait] = 1

    P[c, wait, h] = 1.0
    R[c, wait] = 0

    P[c, search, c] = 1.0
    R[c, search] = 0

    return P, R

def policy_evaluation(P, R, policy, gamma, theta):
    V = np.zeros(S, dtype=float)
    it = 0
    while True:
        delta = 0.0
        V_new = V.copy()
        for s in range(S):
            a = policy[s]
            V_new[s] = R[s, a] + gamma * np.dot(P[s, a], V)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        it += 1
        if delta < theta:
            break
    print(f"Policy evaluation converged in {it} iterations")
    return V

def compute_q(V, P, R, gamma, s, a):
    return R[s, a] + gamma * np.dot(P[s, a], V)

def value_iteration(P, R, gamma, theta):
    V = np.zeros(S, dtype=float)
    history = [V.copy()]
    it = 0
    while True:
        delta = 0.0
        V_new = V.copy()
        for s in range(S):
            qs = [compute_q(V, P, R, gamma, s, a) for a in valid_actions[s]]
            V_new[s] = max(qs)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        history.append(V.copy())
        it += 1
        if delta < theta:
            break
    print(f"Value iteration converged in {it} iterations")
    return V, history

def extract_policy(V_star, P, R, gamma):
    policy = np.zeros(S, dtype=int)
    for s in range(S):
        best_a = max(valid_actions[s], key=lambda a: compute_q(V_star, P, R, gamma, s, a))
        policy[s] = best_a
    return policy

def policy_improvement(V, P, R, gamma, old_policy):
    new_policy = old_policy.copy()
    for s in range(S):
        best_a = max(valid_actions[s], key=lambda a: compute_q(V, P, R, gamma, s, a))
        new_policy[s] = best_a
    stable = np.array_equal(new_policy, old_policy)
    return new_policy, stable

def policy_iteration(P, R, gamma, theta):
    policy = np.array([action_to_idx['Wait']] * S, dtype=int)
    policy[state_to_idx['Charging']] = action_to_idx['Wait']

    value_history = []
    policy_history = []

    step = 0
    while True:
        V = policy_evaluation(P, R, policy, gamma, theta)
        value_history.append(V.copy())
        policy_history.append(policy.copy())

        print(f"Policy iteration step {step}")
        for s in range(S):
            print(f"  {STATES[s]} -> {ACTIONS[policy[s]]}, V={V[s]:.6f}")

        new_policy, stable = policy_improvement(V, P, R, gamma, policy)
        if stable:
            return policy, V, value_history, policy_history
        policy = new_policy
        step += 1

def plot_task1_values(V):
    plt.figure(figsize=(7, 5))
    plt.bar(STATES, V)
    plt.title("Task 1.3: Policy Evaluation Values")
    plt.xlabel("State")
    plt.ylabel("Vπ(s)")
    plt.tight_layout()
    plt.savefig("task1_policy_values.png", dpi=200)
    plt.close()

def plot_value_iteration(history):
    arr = np.array(history)
    plt.figure(figsize=(8, 5))
    for s in range(S):
        plt.plot(arr[:, s], label=STATES[s])
    plt.title("Task 2.3: Value Iteration Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2_value_iteration_convergence.png", dpi=200)
    plt.close()

def plot_policy_iteration(value_history, policy_history):
    vals = np.array(value_history)

    plt.figure(figsize=(8, 5))
    for s in range(S):
        plt.plot(vals[:, s], marker='o', label=STATES[s])
    plt.title("Task 3.3(a): Policy Iteration Values")
    plt.xlabel("Policy Iteration Step")
    plt.ylabel("Vπ(s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task3_policy_iteration_values.png", dpi=200)
    plt.close()

    pol = np.array(policy_history).T
    display = np.vectorize(lambda x: 0 if ACTIONS[x] == 'Search' else 1)(pol)

    plt.figure(figsize=(8, 3.8))
    plt.imshow(display, aspect='auto')
    plt.yticks(range(S), STATES)
    plt.xticks(range(pol.shape[1]), [f"Iter {i}" for i in range(pol.shape[1])])
    plt.title("Task 3.3(b): Policy by Iteration (0=Search, 1=Wait)")
    plt.colorbar()
    for i in range(S):
        for j in range(pol.shape[1]):
            plt.text(j, i, ACTIONS[pol[i, j]], ha='center', va='center')
    plt.tight_layout()
    plt.savefig("task3_policy_iteration_policy.png", dpi=200)
    plt.close()

def write_analysis(V_pi, V_star_vi, optimal_policy_vi, final_policy_pi, final_V_pi, vi_history, pi_value_history):
    def pol_to_lines(policy):
        return "\n".join(f"- {STATES[s]} -> {ACTIONS[policy[s]]}" for s in range(S))

    analysis = f"""Assignment 5 Results

Task 1.2: Policy Evaluation
Policy:
- High -> Search
- Low -> Wait
- Charging -> Wait

Values:
- High: {V_pi[0]:.6f}
- Low: {V_pi[1]:.6f}
- Charging: {V_pi[2]:.6f}

Task 2.2: Optimal Policy from Value Iteration
{pol_to_lines(optimal_policy_vi)}

Optimal Values V*:
- High: {V_star_vi[0]:.6f}
- Low: {V_star_vi[1]:.6f}
- Charging: {V_star_vi[2]:.6f}

Task 3.2: Final Policy from Policy Iteration
{pol_to_lines(final_policy_pi)}

Final Values:
- High: {final_V_pi[0]:.6f}
- Low: {final_V_pi[1]:.6f}
- Charging: {final_V_pi[2]:.6f}

Task 4.1: Compare Value Iteration and Policy Iteration
- Value Iteration needed {len(vi_history)-1} sweeps of the Bellman optimality update.
- Policy Iteration needed {len(pi_value_history)} policy-evaluation phases.
- Policy Iteration typically stabilises in fewer policy changes because each improvement step makes a globally greedy update from the current value estimates.

Task 4.2: Convergence Behavior
- During convergence, all state values rise from 0 and settle to their long-run discounted returns.
- Value Iteration applies a max over actions at every sweep, so it directly moves toward V*.
- Policy Evaluation follows one fixed policy, so it converges to that policy's value function rather than optimising at each step.

Task 4.3: Optimal Policy Interpretation
- High: Search is optimal because it gives a strong immediate reward and still has a high chance to remain in High.
- Low: Search is still optimal here because its expected immediate reward is better than waiting and it may move back to High.
- Charging: Wait is the only allowed action, and it deterministically returns the robot to High.

Task 4.4: Practical Insight
- This policy helps a battery-powered robot balance productivity and energy recovery.
- It formalises when the robot should keep working versus when it should conserve or recharge, which improves long-term performance.

Note on the reward table:
- The assignment asks for R to be shape (3, 2), but the Low/Search row in the transition table gives two different rewards depending on next state.
- To keep R as a 2D array exactly as required, this implementation uses the expected immediate reward for Low/Search:
  R[Low, Search] = 0.4*(-3) + 0.6*(4) = 1.2
"""
    with open("analysis_results.txt", "w", encoding="utf-8") as f:
        f.write(analysis)

def main():
    P, R = build_mdp()

    print("Checking probability sums:")
    for s in range(S):
        for a in range(A):
            print(f"P[{STATES[s]}, {ACTIONS[a]}, :].sum() = {P[s, a].sum():.1f}")

    print("\nReward array R:")
    print(R)

    policy = np.array([
        action_to_idx['Search'],
        action_to_idx['Wait'],
        action_to_idx['Wait']
    ], dtype=int)

    V_pi = policy_evaluation(P, R, policy, GAMMA, THETA)
    print("\nTask 1.2: V_pi")
    for s in range(S):
        print(f"{STATES[s]}: {V_pi[s]:.6f}")
    plot_task1_values(V_pi)

    V_star_vi, vi_history = value_iteration(P, R, GAMMA, THETA)
    optimal_policy_vi = extract_policy(V_star_vi, P, R, GAMMA)
    print("\nTask 2.2: Optimal policy from value iteration")
    for s in range(S):
        print(f"{STATES[s]} -> {ACTIONS[optimal_policy_vi[s]]}")
    plot_value_iteration(vi_history)

    final_policy_pi, final_V_pi, pi_value_history, pi_policy_history = policy_iteration(P, R, GAMMA, THETA)
    print("\nTask 3.2: Final policy from policy iteration")
    for s in range(S):
        print(f"{STATES[s]} -> {ACTIONS[final_policy_pi[s]]}")
    plot_policy_iteration(pi_value_history, pi_policy_history)

    write_analysis(V_pi, V_star_vi, optimal_policy_vi, final_policy_pi, final_V_pi, vi_history, pi_value_history)

if __name__ == "__main__":
    main()
