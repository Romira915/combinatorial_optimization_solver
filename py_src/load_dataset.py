

class KnapsackData:
    def __init__(self, name: str, n: int, capacity: int, opt_value: int, cost_list: list[int], weight_list: list[int], opt_state: list[int]) -> None:
        self.dataset_name = name
        self.n = n
        self.capacity = capacity
        self.optimum_value = opt_value
        self.cost_list = cost_list
        self.weight_list = weight_list
        self.optimum_state = opt_state


def load_knapsack(path: str) -> list[KnapsackData]:
    knapsack_datasets = []
    with open(path) as f:
        lines = []
        while True:
            line = f.readline()
            if line == "":
                break
            line_trim = line.strip()

            if line_trim.startswith("-") and line_trim.endswith("-"):
                knapsack_datasets.append(parse_knapsack_data(lines))
                lines.clear()
                continue

            if line_trim != "":
                lines.append(line_trim)

        return knapsack_datasets


def parse_knapsack_data(data: list[str]) -> KnapsackData:
    dataset_name = data[0]
    n = int(data[1].split(" ")[1])
    c = int(data[2].split(" ")[1])
    opt_value = None
    try:
        opt_value = int(data[3].split(" ")[1])
    except ValueError:
        opt_value = None
    cost_list = []
    weight_list = []
    optimum_state = []

    for i in range(5, n + 5):
        index, cost, weight, selected = data[i].split(",")
        cost_list.append(int(cost))
        weight_list.append(int(weight))
        optimum_state.append(int(selected))

    return KnapsackData(dataset_name, n, c, opt_value, cost_list, weight_list, optimum_state)
