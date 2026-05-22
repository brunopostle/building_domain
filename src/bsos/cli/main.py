import typer
from bsos.cli import extract, validate, curate, review

app = typer.Typer(name="bsos", help="Building Semantic Ontology System", no_args_is_help=True)
app.add_typer(extract.app, name="extract", help="Run extraction pipeline")
app.add_typer(validate.app, name="validate", help="Validate knowledge base")
app.add_typer(curate.app, name="curate", help="Curate entities and predicates")
app.add_typer(review.app, name="review", help="Review pending items")


@app.callback()
def callback() -> None:
    pass


def main() -> None:
    app()
