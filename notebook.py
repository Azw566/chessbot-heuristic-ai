import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import chess
    import chess.svg
    import plotly.graph_objects as go

    from chessbot.genome import Genome, GENE_LABELS
    from chessbot.evaluation import evaluate
    from chessbot.engine import search
    from chessbot.genetic import initialize_population, next_generation
    from chessbot.tournament import run_tournament
    from chessbot.utils import Timer

    return (
        GENE_LABELS,
        Genome,
        Timer,
        chess,
        go,
        initialize_population,
        mo,
        next_generation,
        np,
        run_tournament,
    )


@app.cell
def title(mo):
    mo.md("""
    # Evolutionary ChessBot

    Train a chess engine through genetic evolution. Genomes encode piece values
    and evaluation weights. A population of bots plays tournaments, and the
    fittest survive to breed the next generation.

    After evolution, browse and replay every game from every generation.
    """)
    return


@app.cell
def parameters(mo):
    params_form = (
        mo.md(
            """
            **Evolution Parameters**

            {pop_size}
            {generations}
            {depth}
            {elite_pct}
            {mutation_rate}
            {mutation_mag}
            {seed}
            """
        )
        .batch(
            pop_size=mo.ui.slider(
                start=4, stop=20, step=2, value=6, label="Population size"
            ),
            generations=mo.ui.slider(
                start=1, stop=20, step=1, value=3, label="Generations"
            ),
            depth=mo.ui.slider(
                start=1, stop=4, step=1, value=2, label="Search depth (plies)"
            ),
            elite_pct=mo.ui.slider(
                start=0.1, stop=0.5, step=0.05, value=0.25, label="Elite fraction"
            ),
            mutation_rate=mo.ui.slider(
                start=0.05, stop=0.5, step=0.05, value=0.2, label="Mutation rate"
            ),
            mutation_mag=mo.ui.slider(
                start=0.05, stop=0.5, step=0.05, value=0.1, label="Mutation magnitude"
            ),
            seed=mo.ui.number(
                start=0, stop=9999, value=42, label="Random seed"
            ),
        )
        .form(submit_button_label="Set Parameters")
    )
    params_form
    return (params_form,)


@app.cell
def run_button(mo):
    run_btn = mo.ui.run_button(label="Run Evolution")
    run_btn
    return (run_btn,)


@app.cell
def evolution(
    Genome,
    Timer,
    initialize_population,
    mo,
    next_generation,
    np,
    params_form,
    run_btn,
    run_tournament,
):
    mo.stop(not run_btn.value)
    mo.stop(params_form.value is None, mo.md("**Set parameters first.**"))

    p = params_form.value
    rng = np.random.default_rng(int(p["seed"]))

    pop = initialize_population(int(p["pop_size"]), rng=rng)

    history = []
    best_per_gen = []
    all_games = {}  # gen_index -> list of game records

    with mo.status.progress_bar(
        total=int(p["generations"]),
        title="Evolving...",
        completion_title="Evolution complete!",
    ) as bar:
        for gen in range(int(p["generations"])):
            with Timer(f"Gen {gen}") as t:
                pop, game_records = run_tournament(
                    pop, depth=int(p["depth"]), max_moves=60,
                )

            all_games[gen] = game_records

            fitnesses = [g.fitness for g in pop]
            best = max(pop, key=lambda g: g.fitness)
            best_per_gen.append(best.copy())

            history.append({
                "generation": gen,
                "best_fitness": max(fitnesses),
                "avg_fitness": float(np.mean(fitnesses)),
                "worst_fitness": min(fitnesses),
                "best_genes": best.to_vector(),
                "elapsed": t.elapsed,
            })

            pop = next_generation(
                pop,
                elite_fraction=float(p["elite_pct"]),
                mutation_rate=float(p["mutation_rate"]),
                mutation_magnitude=float(p["mutation_mag"]),
                rng=rng,
            )

            bar.update()

    champion = best_per_gen[-1] if best_per_gen else Genome()

    mo.md(
        f"Evolution finished: **{int(p['generations'])}** generations, "
        f"**{sum(len(g) for g in all_games.values())}** games played, "
        f"champion fitness = **{champion.fitness:.2f}**"
    )
    return all_games, champion, history


@app.cell
def fitness_plot(go, history, mo):
    mo.stop(not history, mo.md("*Run evolution to see fitness plot.*"))

    gens = [h["generation"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=[h["best_fitness"] for h in history],
        mode="lines+markers", name="Best",
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=[h["avg_fitness"] for h in history],
        mode="lines+markers", name="Average",
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=[h["worst_fitness"] for h in history],
        mode="lines+markers", name="Worst",
    ))
    fig.update_layout(
        title="Fitness Over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        template="plotly_white",
    )
    fig
    return


@app.cell
def weight_plot(GENE_LABELS, go, history, mo):
    mo.stop(not history, mo.md("*Run evolution to see weight evolution.*"))

    gens = [h["generation"] for h in history]
    fig = go.Figure()
    for i, label in enumerate(GENE_LABELS):
        fig.add_trace(go.Scatter(
            x=gens,
            y=[h["best_genes"][i] for h in history],
            mode="lines+markers",
            name=label,
        ))
    fig.update_layout(
        title="Best Genome Weights Over Generations",
        xaxis_title="Generation",
        yaxis_title="Value",
        template="plotly_white",
    )
    fig
    return


@app.cell
def genome_table(GENE_LABELS, champion, mo):
    rows = "".join(
        f"| {label} | {value:.4f} |\n"
        for label, value in zip(GENE_LABELS, champion.genes)
    )
    mo.md(
        f"""
    ### Champion Genome (fitness: {champion.fitness:.2f})

    | Gene | Value |
    |------|-------|
    {rows}
        """
    )
    return


@app.cell
def replay_header(mo):
    mo.md("""
    ---
    ## Game Replay Viewer
    Browse and replay every game from the tournament.
    """)
    return


@app.cell
def game_selector(all_games, mo):
    mo.stop(not all_games, mo.md("*Run evolution to browse games.*"))

    # Build game options: "Gen X - Game Y: Bot A vs Bot B (result)"
    gen_options = {f"Generation {g}": g for g in sorted(all_games.keys())}
    gen_dropdown = mo.ui.dropdown(
        options=gen_options,
        value=1,
        label="Generation",
    )
    gen_dropdown
    return (gen_dropdown,)


@app.cell
def game_picker(all_games, gen_dropdown, mo):
    mo.stop(gen_dropdown.value is None)
    gen = gen_dropdown.value
    games = all_games[gen]

    game_labels = {}
    for i, g in enumerate(games):
        result_icon = {"white": "1-0", "black": "0-1", "draw": "1/2"}[g["result"]]
        label = f"Game {i+1}: Bot #{g['white_idx']} vs Bot #{g['black_idx']} ({result_icon}, {g['moves']} moves)"
        game_labels[label] = i

    game_dropdown = mo.ui.dropdown(
        options=game_labels,
        value=0,
        label="Game",
    )
    game_dropdown
    return (game_dropdown,)


@app.cell
def move_slider_cell(all_games, game_dropdown, gen_dropdown, mo):
    mo.stop(gen_dropdown.value is None or game_dropdown.value is None)
    game = all_games[gen_dropdown.value][game_dropdown.value]
    total_moves = game["moves"]

    move_slider = mo.ui.slider(
        start=0,
        stop=max(total_moves, 1),
        step=1,
        value=0,
        label=f"Move (0 = start, {total_moves} = final)",
        full_width=True,
    )
    move_slider
    return (move_slider,)


@app.cell
def board_replay(
    all_games,
    chess,
    game_dropdown,
    gen_dropdown,
    mo,
    move_slider,
):
    mo.stop(
        gen_dropdown.value is None
        or game_dropdown.value is None
        or move_slider.value is None
    )

    game = all_games[gen_dropdown.value][game_dropdown.value]
    move_list = game["move_list"]
    target_move = move_slider.value

    # Rebuild the board up to the selected move
    board = chess.Board()
    san_log = []
    for i, san in enumerate(move_list[:target_move]):
        move = board.parse_san(san)
        board.push(move)
        # Build numbered move list: "1. e4 e5 2. Nf3 ..."
        if i % 2 == 0:
            san_log.append(f"{i // 2 + 1}. {san}")
        else:
            san_log.append(san)

    # Highlight the last move played
    last_move = board.peek() if board.move_stack else None
    svg = chess.svg.board(board, lastmove=last_move, size=420)

    # Status line
    result_str = {"white": "1-0 (White wins)", "black": "0-1 (Black wins)", "draw": "1/2-1/2 (Draw)"}
    status = ""
    if target_move == game["moves"]:
        status = f"**Final: {result_str[game['result']]}**"
    elif board.is_check():
        status = "**Check!**"

    side = "White" if board.turn == chess.WHITE else "Black"
    move_text = " ".join(san_log) if san_log else "(starting position)"

    mo.vstack([
        mo.md(
            f"### Bot #{game['white_idx']} (White) vs Bot #{game['black_idx']} (Black)\n\n"
            f"**{side} to move** | Move {target_move} / {game['moves']}\n\n"
            f"{status}"
        ),
        mo.Html(svg),
        mo.md(f"**Moves:** {move_text}"),
    ])
    return


if __name__ == "__main__":
    app.run()
