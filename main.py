"""
Лабораторная работа № 2, вариант 8.
Назначение студентов на проекты с учётом предпочтений и вместимости команд.

Запуск:
    python -m pip install numpy pandas matplotlib
    python main.py --runs 20 --seed 42 --generations 250
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_STUDENTS = 20
N_PROJECTS = 5
CAPACITIES = np.array([5, 4, 4, 4, 5], dtype=int)
POPULATION_SIZE = 100
DEFAULT_RUNS = 20
DEFAULT_GENERATIONS = 100
DEFAULT_SEED = 42
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.15
PENALTY_COEFFICIENT = 8.0


def load_preferences(path="preferences.csv"):
    """Загружает матрицу предпочтений: строки — студенты, столбцы — проекты."""
    frame = pd.read_csv(path, index_col=0)
    preferences = frame.to_numpy(dtype=float)
    if preferences.shape != (N_STUDENTS, N_PROJECTS):
        raise ValueError(f"Ожидалась матрица {(N_STUDENTS, N_PROJECTS)}, "
                         f"получено {preferences.shape}")
    return frame.index.tolist(), frame.columns.tolist(), preferences


def counts_for(assignment):
    return np.bincount(assignment, minlength=N_PROJECTS)


def is_feasible(assignment):
    """Каждый студент назначен ровно на один проект; вместимость не превышена."""
    return (len(assignment) == N_STUDENTS
            and np.all((assignment >= 0) & (assignment < N_PROJECTS))
            and np.all(counts_for(assignment) <= CAPACITIES))


def preference_score(assignment, preferences):
    return float(sum(preferences[i, project]
                     for i, project in enumerate(assignment)))


def overload(assignment):
    excess = np.maximum(counts_for(assignment) - CAPACITIES, 0)
    return int(excess.sum())


def penalty_fitness(assignment, preferences, penalty=PENALTY_COEFFICIENT):
    """Скор предпочтений минус штраф за превышение вместимости."""
    excess = np.maximum(counts_for(assignment) - CAPACITIES, 0)
    return preference_score(assignment, preferences) - penalty * float(excess @ excess)


def random_assignment(rng):
    """Создаёт допустимое распределение, учитывая вместимости."""
    slots = np.repeat(np.arange(N_PROJECTS), CAPACITIES)
    chosen = rng.choice(slots, size=N_STUDENTS, replace=False)
    return chosen.astype(int)


def repair_assignment(assignment, preferences, rng):
    """Исправляет превышение вместимости, оставляя студентов на хороших проектах."""
    result = assignment.copy()
    while True:
        counts = counts_for(result)
        overfull = np.where(counts > CAPACITIES)[0]
        if len(overfull) == 0:
            break
        project = int(rng.choice(overfull))
        students = np.where(result == project)[0]
        # Сначала переносим студентов, которым этот проект нравится меньше.
        student_scores = preferences[students, project]
        student = int(students[np.argmin(student_scores)])
        available = np.where(counts < CAPACITIES)[0]
        if len(available) == 0:
            # Теоретически недостижимо при данных ёмкостях, но оставляем защиту.
            break
        # Переносим на допустимый проект с максимальным предпочтением.
        target_scores = preferences[student, available]
        best_targets = available[target_scores == target_scores.max()]
        target = int(rng.choice(best_targets))
        result[student] = target
    return result


def tournament(population, scores, rng, size=3):
    ids = rng.choice(len(population), size=size, replace=False)
    return population[int(ids[np.argmax(scores[ids])])].copy()


def crossover(parent1, parent2, rng):
    """Поточечный выбор назначения от одного из родителей для каждого студента."""
    mask = rng.random(N_STUDENTS) < 0.5
    child1 = np.where(mask, parent1, parent2).astype(int)
    child2 = np.where(mask, parent2, parent1).astype(int)
    return child1, child2


def mutate_reassign(assignment, rng, probability=MUTATION_PROBABILITY):
    """Мутация: случайно переназначить выбранных студентов на другие проекты."""
    child = assignment.copy()
    for i in range(N_STUDENTS):
        if rng.random() < probability:
            choices = [p for p in range(N_PROJECTS) if p != child[i]]
            child[i] = int(rng.choice(choices))
    return child


def mutate_swap(assignment, rng, probability=MUTATION_PROBABILITY):
    """Мутация: обменять назначения двух студентов; сохраняет количества по проектам."""
    child = assignment.copy()
    if rng.random() < probability:
        i, j = rng.choice(N_STUDENTS, size=2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child


def genetic_algorithm(preferences, seed, generations=DEFAULT_GENERATIONS,
                      population_size=POPULATION_SIZE, constraint_mode="repair",
                      mutation_mode="reassign"):
    rng = np.random.default_rng(seed)
    population = []
    for _ in range(population_size):
        candidate = random_assignment(rng)
        if constraint_mode == "repair":
            candidate = repair_assignment(candidate, preferences, rng)
        population.append(candidate)
    population = np.asarray(population, dtype=int)

    def score(a):
        if constraint_mode == "repair":
            return preference_score(a, preferences)
        return penalty_fitness(a, preferences)

    scores = np.array([score(a) for a in population])
    history = []
    best_feasible = None
    best_feasible_score = -float("inf")
    for assignment, value in zip(population, scores):
        if is_feasible(assignment):
            raw = preference_score(assignment, preferences)
            if raw > best_feasible_score:
                best_feasible, best_feasible_score = assignment.copy(), raw

    for generation in range(generations):
        new_population = []
        elite_idx = int(np.argmax(scores))
        elite = population[elite_idx].copy()
        if constraint_mode == "repair":
            elite = repair_assignment(elite, preferences, rng)
        new_population.append(elite)

        while len(new_population) < population_size:
            p1 = tournament(population, scores, rng)
            p2 = tournament(population, scores, rng)
            if rng.random() < CROSSOVER_PROBABILITY:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            mutator = mutate_reassign if mutation_mode == "reassign" else mutate_swap
            c1 = mutator(c1, rng)
            c2 = mutator(c2, rng)

            if constraint_mode == "repair":
                c1 = repair_assignment(c1, preferences, rng)
                c2 = repair_assignment(c2, preferences, rng)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = np.asarray(new_population, dtype=int)
        scores = np.array([score(a) for a in population])

        feasible = [a for a in population if is_feasible(a)]
        if feasible:
            candidate = max(feasible, key=lambda a: preference_score(a, preferences))
            value = preference_score(candidate, preferences)
            if value > best_feasible_score:
                best_feasible = candidate.copy()
                best_feasible_score = value

        if best_feasible is not None:
            history.append(best_feasible_score)
        else:
            history.append(float(np.max(scores)))

    if best_feasible is not None:
        best = best_feasible
        best_score = preference_score(best, preferences)
        feasible_result = True
    else:
        best_idx = int(np.argmax(scores))
        best = population[best_idx].copy()
        best_score = preference_score(best, preferences)
        feasible_result = is_feasible(best)

    return best, best_score, history, feasible_result


def random_feasible_search(preferences, seed, evaluations):
    """Случайный поиск по допустимым назначениям с заданным числом оценок."""
    rng = np.random.default_rng(seed)
    best, best_score = None, -float("inf")
    for _ in range(evaluations):
        candidate = random_assignment(rng)
        value = preference_score(candidate, preferences)
        if value > best_score:
            best, best_score = candidate.copy(), value
    return best, best_score


def make_flowchart(path):
    """Строит блок-схему с понятным циклом по поколениям."""
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")

    x = 5.5
    boxes = [
        (12.8, "Начало: загрузить предпочтения\\nи ограничения", 1.2),
        (11.1, "Создать начальную популяцию\\nназначений", 1.2),
        (9.4, "Оценить предпочтения\\nи нарушения вместимости", 1.2),
        (7.7, "Выбрать родителей\\nтурнирным отбором", 1.2),
        (6.0, "Выполнить кроссовер\\nи мутацию назначений", 1.2),
        (4.3, "Применить repair или штраф\\nза нарушение вместимости", 1.2),
    ]
    for y, label, width in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#333", lw=1.2))
    # Vertical arrows between process blocks.
    for (y1, _, _), (y2, _, _) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(x, y2 + 0.48), xytext=(x, y1 - 0.48),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

    # Generation-limit decision.
    diamond = plt.Polygon([(x, 3.1), (x + 1.55, 2.35), (x, 1.6), (x - 1.55, 2.35)],
                          closed=True, fc="white", ec="#333", lw=1.2)
    ax.add_patch(diamond)
    ax.text(x, 2.35, "Достигнут лимит\\nпоколений?", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(x, 3.1), xytext=(x, 3.82),
                arrowprops=dict(arrowstyle="->", lw=1.2))

    # Yes -> output.
    ax.text(x + 1.95, 1.7, "Да", fontsize=9)
    ax.annotate("", xy=(x, 0.55), xytext=(x, 1.6),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(x, 0.25, "Вывести лучшее допустимое назначение\\nи завершить",
            ha="center", va="center", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#333", lw=1.2))

    # No -> return to selection/evolution, routed along the right margin.
    ax.text(8.0, 2.55, "Нет", fontsize=9)
    ax.annotate("", xy=(7.2, 7.7), xytext=(7.05, 2.35),
                arrowprops=dict(arrowstyle="->", lw=1.2,
                                connectionstyle="angle3,angleA=0,angleB=90"))
    ax.text(8.35, 5.2, "Следующее\\nпоколение", fontsize=8.5, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    args = parser.parse_args()
    if args.runs < 20:
        raise ValueError("Требуется не менее 20 запусков.")
    names, projects, preferences = load_preferences()

    configurations = [
        ("GA_repair_reassign", "repair", "reassign"),
        ("GA_penalty_reassign", "penalty", "reassign"),
        ("GA_repair_swap", "repair", "swap"),
    ]
    records = []
    histories = {}
    for label, mode, mutation in configurations:
        histories[label] = []
        for run in range(args.runs):
            seed = args.seed + run
            assignment, score, history, feasible = genetic_algorithm(
                preferences, seed, args.generations,
                constraint_mode=mode, mutation_mode=mutation
            )
            histories[label].append(history)
            records.append({
                "algorithm": label, "run": run + 1, "seed": seed,
                "preference_score": score, "feasible": feasible,
                "overload": overload(assignment),
                "assignment": json.dumps(assignment.tolist())
            })
            print(f"{label}, запуск {run+1:02d}: score={score:.1f}; "
                  f"feasible={feasible}; overload={overload(assignment)}")

    histories["Random_feasible"] = []
    for run in range(args.runs):
        seed = args.seed + run
        assignment, score = random_feasible_search(
            preferences, seed, evaluations=args.generations * POPULATION_SIZE
        )
        # A constant history baseline is sufficient for the same evaluation budget.
        histories["Random_feasible"].append([score] * args.generations)
        records.append({
            "algorithm": "Random_feasible", "run": run + 1, "seed": seed,
            "preference_score": score, "feasible": is_feasible(assignment),
            "overload": overload(assignment),
            "assignment": json.dumps(assignment.tolist())
        })
        print(f"Random_feasible, запуск {run+1:02d}: score={score:.1f}; "
              f"feasible={is_feasible(assignment)}")

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("algorithm").agg(
        best_score=("preference_score", "max"),
        mean_score=("preference_score", "mean"),
        median_score=("preference_score", "median"),
        std_score=("preference_score", "std"),
        worst_score=("preference_score", "min"),
        feasible_rate=("feasible", "mean"),
        mean_overload=("overload", "mean")
    )
    summary.to_csv("summary.csv", encoding="utf-8-sig")
    print("\\nСводная статистика:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    # Plot average best-so-far score per generation.
    plt.figure(figsize=(10, 6))
    for label, curves in histories.items():
        values = np.asarray(curves, dtype=float)
        mean = values.mean(axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        plt.plot(np.arange(1, args.generations+1), mean, label=label)
        plt.fill_between(np.arange(1, args.generations+1), q25, q75, alpha=0.14)
    plt.xlabel("Поколение")
    plt.ylabel("Лучший суммарный балл предпочтений")
    plt.title("Сходимость алгоритмов назначения студентов на проекты")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("convergence.png", dpi=180); plt.close()

    make_flowchart("flowchart.png")
    # Save best feasible assignment found across GA runs.
    feasible_rows = df[df["feasible"]]
    if not feasible_rows.empty:
        row = feasible_rows.loc[feasible_rows["preference_score"].idxmax()]
        assignment = np.array(json.loads(row["assignment"]), dtype=int)
        decoded = []
        for p, project in enumerate(projects):
            members = [names[i] for i, val in enumerate(assignment) if val == p]
            decoded.append({"project": project, "capacity": int(CAPACITIES[p]),
                            "assigned": len(members), "students": ", ".join(members)})
        pd.DataFrame(decoded).to_csv("best_assignment.csv", index=False, encoding="utf-8-sig")
        print("\\nЛучшее допустимое назначение:", row["algorithm"], "score=", row["preference_score"])
        print(pd.DataFrame(decoded).to_string(index=False))
    print("\\nВсе запуски:", len(df), "Статистика сохранена в summary.csv")


if __name__ == "__main__":
    main()
