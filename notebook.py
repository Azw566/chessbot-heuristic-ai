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

    _gens = [h["generation"] for h in history]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_gens, y=[h["best_fitness"] for h in history],
        mode="lines+markers", name="Best",
    ))
    _fig.add_trace(go.Scatter(
        x=_gens, y=[h["avg_fitness"] for h in history],
        mode="lines+markers", name="Average",
    ))
    _fig.add_trace(go.Scatter(
        x=_gens, y=[h["worst_fitness"] for h in history],
        mode="lines+markers", name="Worst",
    ))
    _fig.update_layout(
        title="Fitness Over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        template="plotly_white",
    )
    _fig
    return


@app.cell
def weight_plot(GENE_LABELS, go, history, mo):
    mo.stop(not history, mo.md("*Run evolution to see weight evolution.*"))

    _gens = [h["generation"] for h in history]
    _fig = go.Figure()
    for _i, _label in enumerate(GENE_LABELS):
        _fig.add_trace(go.Scatter(
            x=_gens,
            y=[h["best_genes"][_i] for h in history],
            mode="lines+markers",
            name=_label,
        ))
    _fig.update_layout(
        title="Best Genome Weights Over Generations",
        xaxis_title="Generation",
        yaxis_title="Value",
        template="plotly_white",
    )
    _fig
    return


@app.cell
def genome_table(GENE_LABELS, champion, mo):
    _rows = "".join(
        f"| {_label} | {_value:.4f} |\n"
        for _label, _value in zip(GENE_LABELS, champion.genes)
    )
    mo.md(
        f"""
    ### Champion Genome (fitness: {champion.fitness:.2f})

    | Gene | Value |
    |------|-------|
    {_rows}
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
        value="Generation 0",
        label="Generation",
    )
    gen_dropdown
    return (gen_dropdown,)


@app.cell
def game_picker(all_games, gen_dropdown, mo):
    mo.stop(gen_dropdown.value is None)
    _gen = gen_dropdown.value
    _games = all_games[_gen]

    _game_labels = {}
    for _i, _g in enumerate(_games):
        _result_icon = {"white": "1-0", "black": "0-1", "draw": "1/2"}[_g["result"]]
        _label = f"Game {_i+1}: Bot #{_g['white_idx']} vs Bot #{_g['black_idx']} ({_result_icon}, {_g['moves']} moves)"
        _game_labels[_label] = _i

    game_dropdown = mo.ui.dropdown(
        options=_game_labels,
        value=list(_game_labels.keys())[0] if _game_labels else None,
        label="Game",
    )
    game_dropdown
    return (game_dropdown,)


@app.cell
def move_slider_cell(all_games, game_dropdown, gen_dropdown, mo):
    mo.stop(gen_dropdown.value is None or game_dropdown.value is None)
    _game = all_games[gen_dropdown.value][game_dropdown.value]
    _total_moves = _game["moves"]

    move_slider = mo.ui.slider(
        start=0,
        stop=max(_total_moves, 1),
        step=1,
        value=0,
        label=f"Move (0 = start, {_total_moves} = final)",
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

    _game = all_games[gen_dropdown.value][game_dropdown.value]
    _move_list = _game["move_list"]
    _target_move = move_slider.value

    # Rebuild the board up to the selected move
    _board = chess.Board()
    _san_log = []
    for _i, _san in enumerate(_move_list[:_target_move]):
        _move = _board.parse_san(_san)
        _board.push(_move)
        # Build numbered move list: "1. e4 e5 2. Nf3 ..."
        if _i % 2 == 0:
            _san_log.append(f"{_i // 2 + 1}. {_san}")
        else:
            _san_log.append(_san)

    # Highlight the last move played
    _last_move = _board.peek() if _board.move_stack else None
    _svg = chess.svg.board(_board, lastmove=_last_move, size=420)

    # Status line
    _result_str = {"white": "1-0 (White wins)", "black": "0-1 (Black wins)", "draw": "1/2-1/2 (Draw)"}
    _status = ""
    if _target_move == _game["moves"]:
        _status = f"**Final: {_result_str[_game['result']]}**"
    elif _board.is_check():
        _status = "**Check!**"

    _side = "White" if _board.turn == chess.WHITE else "Black"
    _move_text = " ".join(_san_log) if _san_log else "(starting position)"

    mo.vstack([
        mo.md(
            f"### Bot #{_game['white_idx']} (White) vs Bot #{_game['black_idx']} (Black)\n\n"
            f"**{_side} to move** | Move {_target_move} / {_game['moves']}\n\n"
            f"{_status}"
        ),
        mo.Html(_svg),
        mo.md(f"**Moves:** {_move_text}"),
    ])
    return


if __name__ == "__main__":
    app.run()
