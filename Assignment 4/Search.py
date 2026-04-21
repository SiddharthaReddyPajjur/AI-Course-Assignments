import sys
import heapq
from collections import defaultdict
import itertools

# ==========================================
# SECTION 1: Data Structures
# ==========================================

class Assignment:
    def __init__(self, assign_id, inputs, outcome, food):
        self.assign_id = assign_id
        self.inputs = set(inputs)  # The nodes required before this can be solved
        self.outcome = outcome     # The node this assignment unlocks
        self.food = food           # The food item required (e.g., 'TC', 'PM')

    def __repr__(self):
        return f"A{self.assign_id}"

class SchedulingEnv:
    def __init__(self):
        self.food_costs = {}
        self.group_size = 0
        self.initial_nodes = set()
        self.target_outcomes = set()
        self.assignments = {} # Maps assignment ID to Assignment object

# ==========================================
# SECTION 2: Input Parser
# ==========================================

def parse_input_file(filepath):
    env = SchedulingEnv()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Comments") or line.startswith("Cost") or line.startswith("Group") or line.startswith("Inputs") or line.startswith("Outputs") or line.startswith("Assignment"):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            row_type = parts[0]
            
            # Parse Food Costs
            if row_type == 'C':
                # Handles cases where 'C' is attached to the food like 'CTC 1' or separated like 'C TC 1'
                if len(parts) == 3:
                    env.food_costs[parts[1]] = int(parts[2])
                elif len(parts) == 2:
                    env.food_costs[parts[0][1:]] = int(parts[1])
                    
            # Parse Group Size
            elif row_type == 'G' or row_type.startswith('G'):
                if len(parts) == 2:
                    env.group_size = int(parts[1])
                else:
                    env.group_size = int(parts[0][1:])
                    
            # Parse Initial Inputs
            elif row_type == 'I':
                # Read all numbers until -1
                for val in parts[1:]:
                    if val == '-1':
                        break
                    env.initial_nodes.add(int(val))
                    
            # Parse Target Outcomes
            elif row_type == 'O':
                # Read all numbers until -1
                for val in parts[1:]:
                    if val == '-1':
                        break
                    env.target_outcomes.add(int(val))
                    
            # Parse Assignments
            elif row_type == 'A':
                # Format: A <id> <input1> <input2> <outcome> <Food-name>
                a_id = int(parts[1])
                
                # Extract inputs (all integers before the outcome and food string)
                inputs = []
                idx = 2
                while parts[idx].isdigit():
                    inputs.append(int(parts[idx]))
                    idx += 1
                
                # The last two elements are the outcome node and the food string
                outcome = inputs.pop() 
                food = parts[idx]
                
                env.assignments[a_id] = Assignment(a_id, inputs, outcome, food)

    return env

# ==========================================
# SECTION 3: Cost Calculation Helper
# ==========================================

def calculate_schedule_cost(schedule, food_costs):
    """
    Takes a schedule (list of days, where each day is a list of Assignment objects)
    and calculates the total cost based on the fixed daily menu rule.
    """
    if not schedule:
        return 0, {}

    max_daily_food = defaultdict(int)
    
    # Find the maximum quantity of each food item needed on any single day
    for day in schedule:
        daily_count = defaultdict(int)
        for assignment in day:
            daily_count[assignment.food] += 1
            
        for food, count in daily_count.items():
            max_daily_food[food] = max(max_daily_food[food], count)
            
    # Calculate the cost of this fixed menu
    daily_menu_cost = 0
    for food, count in max_daily_food.items():
        daily_menu_cost += count * food_costs.get(food, 0)
        
    total_days = len(schedule)
    total_cost = total_days * daily_menu_cost
    
    return total_cost, max_daily_food

def calculate_depths(env):
    """
    Calculates the maximum dependency depth (critical path length) for each assignment.
    Returns a dictionary mapping assignment_id -> depth.
    """
    # Build an adjacency list: node -> list of Assignments that require it
    node_to_dependent_assignments = defaultdict(list)
    for a in env.assignments.values():
        for inp in a.inputs:
            node_to_dependent_assignments[inp].append(a)
            
    memo = {}
    
    def get_node_depth(node_id):
        # Return cached result if already computed
        if node_id in memo:
            return memo[node_id]
            
        # Base case: Target outcomes have a depth of 0
        if node_id in env.target_outcomes:
            return 0
            
        dependent_assignments = node_to_dependent_assignments.get(node_id, [])
        if not dependent_assignments:
            return 0 # Dead end (leaf node that isn't a target)
            
        max_depth = 0
        for a in dependent_assignments:
            # The depth of an assignment is 1 + the depth of the node it unlocks
            a_depth = 1 + get_node_depth(a.outcome)
            max_depth = max(max_depth, a_depth)
            
        memo[node_id] = max_depth
        return max_depth
        
    # Calculate and store the depth for every assignment
    assign_depths = {}
    for a in env.assignments.values():
        assign_depths[a.assign_id] = 1 + get_node_depth(a.outcome)
        
    return assign_depths

def calculate_levels(env):
    """
    Calculates the topological level (distance from initial nodes) for each assignment.
    Used for the 'earliest deadline' topological greedy strategy.
    """
    levels = {node: 0 for node in env.initial_nodes}
    
    def get_node_level(node_id):
        if node_id in levels:
            return levels[node_id]
            
        # Find which assignment produces this node
        for a in env.assignments.values():
            if a.outcome == node_id:
                # The level of an outcome is 1 + the max level of its inputs
                max_input_level = max(get_node_level(inp) for inp in a.inputs) if a.inputs else 0
                levels[node_id] = max_input_level + 1
                return levels[node_id]
        return 0
        
    assign_levels = {}
    for a in env.assignments.values():
        assign_levels[a.assign_id] = max(get_node_level(inp) for inp in a.inputs) if a.inputs else 0
        
    return assign_levels

# ==========================================
# SECTION 4: Task 1 - Greedy Algorithms
# ==========================================
def greedy_schedule(env, strategy="cost"):
    """
    Generates a valid schedule using a greedy approach.
    """
    completed_nodes = set(env.initial_nodes)
    remaining_assignments = list(env.assignments.values())
    schedule = [] 
    
    while not env.target_outcomes.issubset(completed_nodes):
        available = [a for a in remaining_assignments if a.inputs.issubset(completed_nodes)]
                
        if not available:
            print("Error: Deadlock reached. Cannot unlock further nodes.")
            break

        # --- THE 4-WAY STRATEGY ROUTER ---
        if strategy == "cost":
            # 1. Cheapest Food: Sort ascending by food cost [cite: 79]
            available.sort(key=lambda a: (env.food_costs.get(a.food, float('inf')), a.assign_id))
            
        elif strategy == "depth":
            # 2. Critical Path: Sort descending by depth (distance to end) [cite: 80]
            if not hasattr(env, 'depths'):
                env.depths = calculate_depths(env)
            available.sort(key=lambda a: (-env.depths[a.assign_id], a.assign_id))
            
        elif strategy == "frequency":
            # 3. Max Frequency: Sort descending by how often the food appears in remaining tasks 
            food_counts = defaultdict(int)
            for a in remaining_assignments:
                food_counts[a.food] += 1
            available.sort(key=lambda a: (-food_counts[a.food], a.assign_id))
            
        elif strategy == "topo":
            # 4. Topological Order: Sort ascending by level (distance from start) [cite: 82]
            if not hasattr(env, 'levels'):
                env.levels = calculate_levels(env)
            available.sort(key=lambda a: (env.levels[a.assign_id], a.assign_id))
            
        # ---------------------------------
            
        selected_for_day = available[:env.group_size]
        schedule.append(selected_for_day)
        
        for a in selected_for_day:
            completed_nodes.add(a.outcome)
            remaining_assignments.remove(a)
            
    return schedule

# ==========================================
# SECTION 5: Task 2 - A* Search (UPDATED)
# ==========================================
def astar_schedule(env):
    """
    A* Search to find the schedule that minimizes total fixed food cost.
    """
    class State:
        def __init__(self, schedule, completed, remaining, g, h, max_menu):
            self.schedule = schedule   
            self.completed = completed 
            self.remaining = remaining 
            self.g = g                 
            self.h = h                 
            self.f = g + h
            self.max_menu = max_menu # Track the menu bloat for this specific path
            
        def __lt__(self, other):
            if self.f != other.f: return self.f < other.f
            if len(self.completed) != len(other.completed): return len(self.completed) > len(other.completed)
            return len(self.schedule) < len(other.schedule)

    min_food_cost = min(env.food_costs.values()) if env.food_costs else 0
    def get_h(completed_nodes):
        unmet_targets = env.target_outcomes - completed_nodes
        return len(unmet_targets) * min_food_cost if unmet_targets else 0

    initial_completed = frozenset(env.initial_nodes)
    initial_remaining = frozenset(env.assignments.values())
    
    # Start with an empty max menu
    start_state = State([], initial_completed, initial_remaining, 0, get_h(initial_completed), {})
    
    open_list = [start_state]
    visited = {} 
    states_explored = 0 
    
    while open_list:
        current = heapq.heappop(open_list)
        states_explored += 1
        
        if env.target_outcomes.issubset(current.completed):
            print(f"Total number of states explored: {states_explored}")
            return [list(day) for day in current.schedule]
            
        # Pruning: We must include the max_menu in the key, because a state with the 
        # same completed nodes but a smaller fixed menu is mathematically better!
        state_key = (current.completed, frozenset(current.max_menu.items()))
        if state_key in visited and visited[state_key] <= current.g:
            continue
        visited[state_key] = current.g
        
        available = [a for a in current.remaining if a.inputs.issubset(current.completed)]
        if not available: continue
            
        max_r = min(len(available), env.group_size)
        valid_daily_combos = []
        for r in range(1, max_r + 1):
            valid_daily_combos.extend(itertools.combinations(available, r))
            
        for combo in valid_daily_combos:
            new_schedule = current.schedule + [combo]
            
            # 1. Calculate the new max daily menu for this proposed step
            new_max_menu = dict(current.max_menu)
            combo_counts = {}
            for a in combo:
                combo_counts[a.food] = combo_counts.get(a.food, 0) + 1
            
            for f, c in combo_counts.items():
                new_max_menu[f] = max(new_max_menu.get(f, 0), c)
                
            # 2. Calculate the TRUE g(n): Fixed Menu Cost * Total Days so far
            fixed_menu_cost = sum(count * env.food_costs.get(food, 0) for food, count in new_max_menu.items())
            new_g = fixed_menu_cost * len(new_schedule)
            
            new_completed = current.completed.union(a.outcome for a in combo)
            new_state = State(
                new_schedule, 
                new_completed, 
                current.remaining.difference(combo), 
                new_g, 
                get_h(new_completed),
                new_max_menu
            )
            heapq.heappush(open_list, new_state)
            
    return []
# ==========================================
# SECTION 5.5: Output Formatter
# ==========================================
def print_schedule(schedule, env, strategy_name):
    print(f"Strategy: {strategy_name}")
    
    # 1. Calculate the Fixed Daily Menu (Global Maximums)
    max_daily_food = {}
    for day in schedule:
        daily_count = {}
        for a in day:
            daily_count[a.food] = daily_count.get(a.food, 0) + 1
            
        for food, count in daily_count.items():
            max_daily_food[food] = max(max_daily_food.get(food, 0), count)
            
    # Calculate the cost of this fixed menu per day
    fixed_menu_cost = sum(count * env.food_costs.get(food, 0) for food, count in max_daily_food.items())
    total_days = len(schedule)
    total_fixed_cost = total_days * fixed_menu_cost

    # 2. Print Day-by-Day Breakdown
    for i, day in enumerate(schedule):
        assign_strs = [f"A{a.assign_id}" for a in day]
        print(f"Day-{i+1}: {', '.join(assign_strs)}")
        
        daily_counts = {}
        for a in day:
            daily_counts[a.food] = daily_counts.get(a.food, 0) + 1
            
        menu_strs = [f"{count}-{food}" for food, count in daily_counts.items()]
        print(f"Consumed: {', '.join(menu_strs)}")
        
        # (Optional) Print what the day would cost individually
        daily_cost = sum(count * env.food_costs.get(food, 0) for food, count in daily_counts.items())
        print(f"Daily Individual Cost: {daily_cost}")

    # 3. Print the Final Summary Requirements
    print(f"Total Days: {total_days}")
    
    fixed_menu_strs = [f"{count}-{food}" for food, count in max_daily_food.items()]
    print(f"Fixed Daily Menu: <{', '.join(fixed_menu_strs)}>")
    print(f"Total Food Cost: {total_fixed_cost}")
    print("-" * 30)

# ==========================================
# SECTION 6: Main Execution
# ==========================================
if __name__ == "__main__":
    
    # 1. SETUP THE ENVIRONMENT
    filename = 'sample_input.txt' # Make sure to change this to your actual file!
    print(f"--- Loading data from {filename} ---")
    
    try:
        env = parse_input_file(filename) 
    except FileNotFoundError:
        print(f"Could not find {filename}. Please ensure it is in the same folder.")
        sys.exit()
    
    # 2. RUN ALL 4 GREEDY STRATEGIES
    strategies = {
        "cost": "Greedy by Food Cost",
        "depth": "Greedy by Dependency Depth",
        "frequency": "Greedy by Food Type Frequency",
        "topo": "Greedy by Earliest Deadline (Topo)"
    }
    
    best_greedy_cost = float('inf')
    best_greedy_name = ""
    best_greedy_schedule = []
    
    for strat_code, strat_name in strategies.items():
        print(f"\n--- Running {strat_name} ---")
        schedule = greedy_schedule(env, strategy=strat_code)
        print_schedule(schedule, env, strat_name)
        
        cost, _ = calculate_schedule_cost(schedule, env.food_costs)
        
        # Track the best performing greedy strategy
        if cost < best_greedy_cost:
            best_greedy_cost = cost
            best_greedy_name = strat_name
            best_greedy_schedule = schedule

    # 3. RUN A* SEARCH
    print("\n--- Running A* Search ---")
    optimal_schedule = astar_schedule(env)
    print_schedule(optimal_schedule, env, "A* Optimal")
    
    # 4. FINAL COMPARISON
    astar_cost_total, _ = calculate_schedule_cost(optimal_schedule, env.food_costs)
    
    cost_diff = best_greedy_cost - astar_cost_total
    day_diff = len(best_greedy_schedule) - len(optimal_schedule)
    
    print(f"--- Comparison: Best Greedy ({best_greedy_name}) vs A* ---")
    print(f"A* Cost: {astar_cost_total} vs Best Greedy Cost: {best_greedy_cost}")
    print(f"Cost Difference: {cost_diff} (A* saved {cost_diff} cost)")
    print(f"Day Difference: {day_diff} (A* used {-day_diff if day_diff < 0 else day_diff} {'more' if day_diff < 0 else 'fewer'} days)")
