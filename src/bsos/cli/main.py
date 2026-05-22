import typer
from bsos.cli import extract, validate, curate, review, config
from bsos.cli.init import app as init_app

app = typer.Typer(name="bsos", help="Building Semantic Ontology System", no_args_is_help=True)

app.add_typer(init_app, name="init", help="Initialise the BSOS database")
app.add_typer(extract.app, name="extract", help="Run extraction pipeline")
app.add_typer(validate.app, name="validate", help="Validate knowledge base")
app.add_typer(curate.app, name="curate", help="Curate entities and predicates")
app.add_typer(review.app, name="review", help="Review pending items")
app.add_typer(config.app, name="config", help="Manage runtime configuration")


@app.callback()
def callback() -> None:
    pass


def main() -> None:
    app()
