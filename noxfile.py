"""Nox sessions."""

import shutil
from pathlib import Path
from textwrap import dedent

import nox
from nox import Session, session

package = "inscar"
owner, repository = "engeir", "inscar"
python_versions = ["3.9", "3.10", "3.11", "3.12"]
nox.options.default_venv_backend = "uv"
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-build",
)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the session's
    virtual environment. This allows pre-commit to locate hooks in that environment when
    invoked from git.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    if session.bin is None:
        return

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes
        if not (
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
        ):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(name="pre-commit", python="3.12")
def precommit(session: Session) -> None:
    """Lint using pre-commit.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install("-e", ".")
    session.install(
        "ruff",
        "pydocstringformatter",
        "mypy",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pytest",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    args = session.posargs or ["src", "tests"]
    session.install("-e", ".")
    session.install("mypy", "pytest")
    session.install("types-attrs")
    session.run("mypy", *args)
    session.run("mypy", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    session.install("-e", ".")
    session.install("coverage[toml]", "pytest", "pygments")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage")


@session(python="3.12")
def coverage(session: Session) -> None:
    """Produce the coverage report.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    session.install("coverage[toml]", "codecov")
    session.run("coverage", "combine")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@session(python=python_versions)
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    session.install("-e", ".")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    args = session.posargs or ["all"]
    session.install("-e", ".")
    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", package, *args)


@session(name="docs-build", python="3.9")
def docs_build(session: Session) -> None:
    """Build the documentation.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    args = session.posargs or ["docs", "docs/_build"]
    session.install("-r", "requirements-dev.lock")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-apidoc", "-o", "docs", "src")
    session.run("sphinx-build", *args)


@session(python="3.12")
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes.

    Parameters
    ----------
    session : Session
        The Session object.
    """
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install("-r", "requirements-dev.lock")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-apidoc", "-o", "docs", "src")
    session.run("sphinx-autobuild", *args)
