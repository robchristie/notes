from __future__ import annotations
import os
from typing import List, Literal
from pydantic import BaseModel
from tree_sitter_languages import get_language, get_parser

class RepoQueryIn(BaseModel):
    kind: Literal["symbol","path","imports","test_of"]
    query: str

class SymbolRef(BaseModel):
    file: str
    line: int
    symbol: str

class RepoQueryOut(BaseModel):
    results: List[SymbolRef]

def _index_python_symbol_nodes(source: bytes, tree, path: str) -> List[SymbolRef]:
    # naive TS walker for python: capture def/class names and import names
    results: List[SymbolRef] = []
    cursor = tree.walk()
    def node_text(n): return source[n.start_byte:n.end_byte].decode(errors="ignore")
    def push(n, sym):
        results.append(SymbolRef(file=path, line=n.start_point[0]+1, symbol=sym))

    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        for c in n.children:
            stack.append(c)
        if n.type in ("function_definition","class_definition"):
            # child[1] should be identifier
            for c in n.children:
                if c.type == "identifier":
                    push(n, node_text(c))
        if n.type in ("import_from_statement","import_statement"):
            push(n, node_text(n))
    return results

async def repo_query(inp: RepoQueryIn, root: str | None = None) -> RepoQueryOut:
    root = root or os.getcwd()
    parser = get_parser("python")
    results: List[SymbolRef] = []
    if inp.kind == "path":
        p = os.path.join(root, inp.query)
        if os.path.exists(p):
            results.append(SymbolRef(file=p, line=1, symbol=inp.query))
        return RepoQueryOut(results=results)

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                src = open(path, "rb").read()
                tree = parser.parse(src)
                symbols = _index_python_symbol_nodes(src, tree, path)
                if inp.kind == "symbol":
                    results.extend([s for s in symbols if inp.query in s.symbol])
                elif inp.kind == "imports":
                    results.extend([s for s in symbols if s.symbol.startswith("import") and inp.query in s.symbol])
                elif inp.kind == "test_of":
                    # heuristic: return tests that reference query in filename
                    if "test" in os.path.basename(path) and inp.query.split(".")[0] in open(path, "r", encoding="utf-8", errors="ignore").read():
                        results.append(SymbolRef(file=path, line=1, symbol="test"))
            except Exception:
                pass
    return RepoQueryOut(results=results)
