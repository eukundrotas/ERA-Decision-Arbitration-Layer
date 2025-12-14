#!/usr/bin/env python3
"""
ERA Decision & Arbitration Layer — CLI Entry Point
"""

import click
import logging
from pathlib import Path
from src.config import config, Config
from src.orchestrator import MultiLLMOrchestrator
from src.utils import setup_logging, ensure_output_dir, csv_to_xlsx

setup_logging(config.log_level)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--pool', default='science', type=click.Choice(['science', 'math', 'med', 'econ']),
              help='Solver pool')
@click.option('--repeats', default=5, type=int, help='Number of runs for stability')
@click.option('--consensus-topk', default=3, type=int, help='Top-K for consensus (2 or 3)')
@click.option('--epsilon', default=0.07, type=float, help='Gap threshold for consensus')
@click.option('--rebuttal/--no-rebuttal', default=True, help='Enable rebuttal round')
@click.option('--hard-only/--no-hard-only', default=False, help='Only hard select, no consensus')
@click.option('--problem', default=None, type=str, help='Single problem string')
@click.option('--problems-file', default=None, type=click.Path(exists=True),
              help='File with problems (one per line)')
@click.option('--out-dir', default='./out', type=click.Path(),
              help='Output directory for artifacts')
def main(pool, repeats, consensus_topk, epsilon, rebuttal, hard_only, problem, problems_file, out_dir):
    """
    ERA Decision & Arbitration Layer
    
    Reliable decision-making through ensemble LLMs, arbitration, consensus, and rebuttal.
    """
    
    logger.info("=" * 60)
    logger.info("ERA Decision & Arbitration Layer")
    logger.info("=" * 60)
    logger.info(f"Pool: {pool}")
    logger.info(f"Repeats: {repeats}")
    logger.info(f"Consensus topk: {consensus_topk}")
    logger.info(f"Epsilon: {epsilon}")
    logger.info(f"Rebuttal: {rebuttal}")
    logger.info(f"Output dir: {out_dir}")
    
    ensure_output_dir(out_dir)
    
    # Parse problems
    problems = []
    if problem:
        problems.append(("task_001", problem))
    elif problems_file:
        with open(problems_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    problems.append((f"task_{i:03d}", line))
    else:
        click.echo("Error: provide --problem or --problems-file")
        return
    
    logger.info(f"Tasks: {len(problems)}")
    
    # Validate config
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"Configuration error: {e}")
        click.echo("Please set OPENROUTER_API_KEY in .env file or environment")
        return
    
    # Initialize orchestrator
    orchestrator = MultiLLMOrchestrator(
        pool_name=pool,
        domain=pool,
        out_dir=out_dir
    )
    
    # Run
    for task_id, prob_text in problems:
        try:
            result = orchestrator.run_multi(
                problem=prob_text,
                task_id=task_id,
                repeats=repeats,
                consensus_topk=consensus_topk if not hard_only else 1,
                epsilon=epsilon if not hard_only else 0.0,
                enable_rebuttal=rebuttal and not hard_only
            )
            
            click.echo(f"\n✓ Task {task_id} completed")
            click.echo(f"  Decision: {result['decision_mode']}")
            click.echo(f"  Stability: {result['stability']['majority_rate']:.1%}")
            click.echo(f"  Answer: {result['final_answer'][:80]}...")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            click.echo(f"\n✗ Task {task_id} failed: {e}")
    
    # Convert CSV to XLSX
    try:
        csv_to_xlsx(out_dir)
        click.echo(f"\n✓ Artifacts saved to {out_dir}")
        click.echo(f"  - runs.csv")
        click.echo(f"  - runs.xlsx")
        click.echo(f"  - final.json")
        click.echo(f"  - model_quality.json")
    except Exception as e:
        logger.warning(f"Could not create XLSX: {e}")
        click.echo(f"  - runs.csv")
        click.echo(f"  - final.json")
        click.echo(f"  - model_quality.json")

if __name__ == '__main__':
    main()
