# Evolutionary ChessBot

A chess engine that evolves through genetic algorithms. Bots with different evaluation weights compete in tournaments -- the fittest survive and breed the next generation.

## Setup

```bash
pip install -e ".[dev]"
```

## Launch the Notebook

```bash
marimo edit notebook.py
```

This opens the interactive UI in your browser.

## How to Train and Watch Games

### 1. Set Parameters

At the top of the notebook you'll see the **Evolution Parameters** form. Adjust these to your liking:

| Parameter | What it does | Recommended start |
|---|---|---|
| **Population size** | Number of bots per generation | 6 |
| **Generations** | How many rounds of evolution to run | 3-5 |
| **Search depth** | How many moves ahead each bot looks (higher = slower but smarter) | 2 |
| **Elite fraction** | Top % of bots that survive to the next generation unchanged | 0.25 |
| **Mutation rate** | Chance each gene gets randomly tweaked in offspring | 0.2 |
| **Mutation magnitude** | How much a mutated gene changes | 0.1 |
| **Seed** | Random seed for reproducibility | 42 |

Click **Set Parameters** to lock them in.

> **Tip:** Start small (pop=4, gen=2, depth=2) for a quick test run under 2 minutes. Scale up once you're comfortable.

### 2. Run Evolution

Click the **Run Evolution** button. A progress bar shows which generation is being processed.

During each generation:
1. Every bot plays against every other bot (round-robin for small populations)
2. Wins score 3 points, draws score 1, with a speed bonus for quick victories
3. The top bots survive, the rest are replaced by offspring (crossover + mutation)

When it finishes you'll see a summary like:
> Evolution finished: **3** generations, **45** games played, champion fitness = **12.50**

### 3. View the Results

Three panels appear after evolution:

- **Fitness plot** -- Line chart showing best / average / worst fitness per generation. An upward trend means your bots are improving.
- **Weight evolution plot** -- How the champion's 10 genes (piece values + evaluation weights) change over generations.
- **Champion genome table** -- The exact weight values of the current best bot.

### 4. Replay Games

Scroll down to the **Game Replay Viewer** section:

1. **Pick a generation** from the dropdown
2. **Pick a game** -- each entry shows the matchup (e.g. "Bot #0 vs Bot #3") and the result (1-0, 0-1, or 1/2)
3. **Drag the move slider** to step through the game move by move
   - Position 0 = starting position
   - Final position = end of game
4. The board updates live with the last move highlighted, and the full move notation is shown below

## Running Tests

```bash
pytest tests/
```

## Project Structure

```
chessbot/          Core logic (importable Python package)
  genome.py        Bot DNA: 5 piece values + 5 evaluation weights
  evaluation.py    Board scoring: material, mobility, center, king safety, pawns
  engine.py        Minimax search with alpha-beta pruning
  genetic.py       Selection, crossover, mutation
  tournament.py    Round-robin matchmaking and scoring
  utils.py         Save/load populations, timer

tests/             Unit tests for each module
notebook.py        Marimo notebook (the UI you interact with)
```
