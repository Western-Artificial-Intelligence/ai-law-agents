"""CLI interface for Token Budget Auditor."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from . import TokenBudgetAuditor, BudgetUsage


def format_budget(budget: BudgetUsage) -> str:
    """Format budget information for display."""
    return (
        f"Key: {budget.key}\n"
        f"  Tokens: {budget.used_tokens:,} / {budget.max_tokens:,} ({budget.remaining:,} remaining)\n"
        f"  Usage: {budget.used_tokens / budget.max_tokens:.1%} of budget\n"
        f"  Last Updated: {budget.last_updated}"
    )


@click.group()
@click.option(
    "--storage",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=Path("~/.config/bailiff/token_budgets.json").expanduser(),
    help="Path to budget storage file"
)
@click.pass_context
def cli(ctx: click.Context, storage: Path) -> None:
    """Manage token budgets for AI model usage."""
    ctx.ensure_object(dict)
    ctx.obj["auditor"] = TokenBudgetAuditor(storage_path=storage)


@cli.command()
@click.argument("key")
@click.argument("max_tokens", type=int)
@click.option("--metadata", type=click.Path(exists=True), help="JSON file with metadata")
@click.pass_context
def create(ctx: click.Context, key: str, max_tokens: int, metadata: Optional[str]) -> None:
    """Create a new token budget."""
    auditor = ctx.obj["auditor"]
    
    try:
        metadata_dict = {}
        if metadata:
            with open(metadata) as f:
                metadata_dict = json.load(f)
                
        budget = auditor.register_budget(key, max_tokens, metadata_dict)
        click.echo(f"Created budget for {key} with {max_tokens:,} tokens")
        click.echo(format_budget(budget))
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("key")
@click.argument("tokens", type=int)
@click.pass_context
def use(ctx: click.Context, key: str, tokens: int) -> None:
    """Use tokens from a budget."""
    auditor = ctx.obj["auditor"]
    
    try:
        success = auditor.use_tokens(key, tokens)
        if not success:
            click.echo(f"Error: Not enough tokens in budget for {key}", err=True)
            sys.exit(1)
            
        budget = auditor.get_budget(key)
        click.echo(f"Used {tokens:,} tokens from {key}")
        click.echo(format_budget(budget))
        
    except KeyError:
        click.echo(f"Error: No budget found for key: {key}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("key")
@click.pass_context
def show(ctx: click.Context, key: str) -> None:
    """Show budget details."""
    auditor = ctx.obj["auditor"]
    
    budget = auditor.get_budget(key)
    if not budget:
        click.echo(f"Error: No budget found for key: {key}", err=True)
        sys.exit(1)
        
    click.echo(format_budget(budget))


@cli.command()
@click.pass_context
def list_budgets(ctx: click.Context) -> None:
    """List all budgets."""
    auditor = ctx.obj["auditor"]
    
    budgets = auditor.get_usage_summary()
    if not budgets:
        click.echo("No budgets found")
        return
        
    for budget in budgets:
        click.echo(f"{budget['key']}: {budget['used_tokens']:,}/{budget['max_tokens']:,} "
                  f"({budget['used_tokens']/budget['max_tokens']:.1%} used)")


if __name__ == "__main__":
    cli()
