"""
Checks the package against the Google Python Style Guide, §3.8.

Every module, class, function and method has a docstring. Its summary is
one line of at most 80 characters, ends with a period, and is followed by a
blank line if more follows. A function's summary is descriptive ("Returns
…"); a property's reads as an attribute. Functions list every parameter
under `Args:`, document a value under `Returns:` (or `Yields:`), and name
what they raise on purpose under `Raises:`. Classes list their public
attributes under `Attributes:`. Module-level names are CAPS and documented
by a docstring right after them.

Layout: a docstring of more than one line opens with a line break after
its quotes and closes with its quotes on a line of their own (PEP 257
allows the summary on the line after the quotes); a one-line docstring
stays on one line.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "obfuscation_immunity"
"""The package under check."""
FILES = sorted(PACKAGE.rglob("*.py"))
"""Every Python file in it."""
LINE_LIMIT = 80
"""Longest allowed line in a docstring, as in the code (black)."""


def summary_problems(where: str, docstring: str) -> list[str]:
    """
    Finds what is wrong with a docstring's summary line.

    Args:
        where: Names the docstring's owner in the messages.
        docstring: The docstring, as `ast.get_docstring` returns it.

    Returns:
        One message per problem; empty if the summary is right.
    """
    lines = docstring.splitlines()
    if not lines or not lines[0].strip():
        return [f"{where}: empty docstring"]
    problems = []
    if not lines[0].endswith("."):
        problems.append(f"{where}: summary does not end with a period")
    if len(lines) > 1 and lines[1].strip():
        problems.append(f"{where}: summary is not one line then a blank")
    return problems


def section_names(docstring: str, section: str) -> set[str]:
    """
    Lists the names a docstring documents under one section.

    Args:
        docstring: The docstring to read.
        section: The section's title, such as "Args".

    Returns:
        The names given at the start of the section's entries.
    """
    names, inside = set(), False
    for line in docstring.splitlines():
        if line.strip() == f"{section}:":
            inside = True
        elif inside and line and not line.startswith(" "):
            inside = False
        elif inside and line.startswith("    ") and line[4] != " ":
            names.add(line.strip().split(":")[0])
    return names


def parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """
    Lists a function's parameters as `Args:` should name them.

    Args:
        function: The function's node.

    Returns:
        Its parameters, without `self` and `cls`, with `*` and `**` kept.
    """
    arguments = function.args
    named = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    names = [a.arg for a in named if a.arg not in ("self", "cls")]
    if arguments.vararg:
        names.append(f"*{arguments.vararg.arg}")
    if arguments.kwarg:
        names.append(f"**{arguments.kwarg.arg}")
    return names


def function_problems(
    where: str, function: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[str]:
    """
    Finds what a function's docstring misses or gets wrong.

    Args:
        where: Names the function in the messages.
        function: The function's node.

    Returns:
        One message per problem; empty if the docstring is right.
    """
    docstring = ast.get_docstring(function)
    if docstring is None:
        return [f"{where}: no docstring"]
    problems = summary_problems(where, docstring)
    decorators = {ast.unparse(d) for d in function.decorator_list}
    first_word = docstring.split()[0]
    if "property" in decorators:
        if first_word in ("Returns", "Return"):
            problems.append(f"{where}: a property reads as an attribute")
        return problems
    if not first_word.endswith("s"):
        problems.append(f"{where}: summary is not descriptive ('Returns')")
    missing = set(parameters(function)) - section_names(docstring, "Args")
    if missing:
        problems.append(f"{where}: Args misses {sorted(missing)}")
    body = list(ast.walk(function))
    yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in body)
    returns = function.returns is not None and ast.unparse(
        function.returns
    ) not in ("None", "'None'")
    if yields and "Yields:" not in docstring:
        problems.append(f"{where}: no Yields")
    if returns and not yields and "Returns:" not in docstring:
        problems.append(f"{where}: no Returns")
    raises = any(isinstance(n, ast.Raise) and n.exc for n in body)
    if raises and "Raises:" not in docstring:
        problems.append(f"{where}: no Raises")
    return problems


def class_problems(where: str, node: ast.ClassDef) -> list[str]:
    """
    Finds what a class's docstring misses or gets wrong.

    Args:
        where: Names the class in the messages.
        node: The class's node.

    Returns:
        One message per problem; empty if the docstring is right.
    """
    docstring = ast.get_docstring(node)
    if docstring is None:
        return [f"{where}: no docstring"]
    problems = summary_problems(where, docstring)
    public = {
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and not statement.target.id.startswith("_")
        and "ClassVar" not in ast.unparse(statement.annotation)
    }
    missing = public - section_names(docstring, "Attributes")
    if missing:
        problems.append(f"{where}: Attributes misses {sorted(missing)}")
    return problems


def is_docstring(statement: ast.stmt) -> bool:
    """
    Tells whether a statement is a bare string, as a docstring is.

    Args:
        statement: Any statement.

    Returns:
        True if it is an expression made of one string constant.
    """
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def global_problems(path: Path, tree: ast.Module) -> list[str]:
    """
    Finds module-level names that are not CAPS or not documented.

    Args:
        path: The module's file, for the messages.
        tree: The module's node.

    Returns:
        One message per problem; empty if every global is right.
    """
    problems = []
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        following = tree.body[index + 1 : index + 2]
        documented = bool(following) and is_docstring(following[0])
        for target in targets:
            name = ast.unparse(target)
            if name == "__all__":
                continue
            where = f"{path.name}:{statement.lineno} {name}"
            if not name.isupper():
                problems.append(f"{where}: module-level name is not CAPS")
            if not documented:
                problems.append(f"{where}: no docstring after it")
    return problems


def layout_problems(path: Path, tree: ast.Module) -> list[str]:
    """
    Finds docstrings that are too wide or that break the quotes' layout.

    Args:
        path: The module's file.
        tree: The module's node.

    Returns:
        One message per problem; empty if every docstring is laid out right.
    """
    source = path.read_text().splitlines()
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Expr) and is_docstring(node)):
            continue
        assert isinstance(node.value, ast.Constant)
        where = f"{path.name}:{node.lineno}"
        text = str(node.value.value)
        if "\n" in text.strip():
            if not text.startswith("\n"):
                problems.append(f"{where}: no line break after the quotes")
            if text.split("\n")[-1].strip():
                problems.append(f"{where}: closing quotes not on own line")
        for number in range(node.lineno, (node.end_lineno or 0) + 1):
            if len(source[number - 1]) > LINE_LIMIT:
                problems.append(f"{path.name}:{number}: line too long")
    return problems


def file_problems(path: Path) -> list[str]:
    """
    Finds every departure from the standard in one file.

    Args:
        path: The module's file.

    Returns:
        One message per problem; empty if the file follows the standard.
    """
    tree = ast.parse(path.read_text())
    module = ast.get_docstring(tree)
    if module is None:
        problems = [f"{path.name}: no module docstring"]
    else:
        problems = summary_problems(path.name, module)
    problems += global_problems(path, tree)
    problems += layout_problems(path, tree)
    for node in ast.walk(tree):
        where = f"{path.name}:{getattr(node, 'lineno', 0)} "
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            problems += function_problems(where + node.name, node)
        elif isinstance(node, ast.ClassDef):
            problems += class_problems(where + node.name, node)
    return problems


@pytest.mark.parametrize(
    "path", FILES, ids=lambda path: str(path.relative_to(PACKAGE))
)
def test_the_file_follows_the_docstring_standard(path):
    """
    Checks one file of the package against the standard.

    Args:
        path: The file.
    """
    assert file_problems(path) == []
