from .orchestration.client_cli import app

if __name__ == "__main__":
    import typer
    typer.run(app)
