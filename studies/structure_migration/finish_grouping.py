"""Mechanical routing of result writers and common experiment/statistic adapters."""
from pathlib import Path
import ast
import json
import re

ROOT = Path(__file__).resolve().parents[2]


def route_writers(path):
    source=path.read_text(encoding="utf-8")
    offsets=[0]
    for line in source.splitlines(keepends=True): offsets.append(offsets[-1]+len(line))
    def position(line,col):
        # AST offsets are UTF-8 bytes, not Unicode character offsets.
        text=source.splitlines(keepends=True)[line-1]
        return offsets[line-1]+len(text.encode("utf-8")[:col].decode("utf-8"))
    edits=[]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute): continue
        if node.func.attr=="to_csv":
            parent_function=next((f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef) and f.name=='write_csv'),None)
            if parent_function and parent_function.lineno<=node.lineno<=parent_function.end_lineno: continue
            owner=ast.get_source_segment(source,node.func.value)
            call=ast.get_source_segment(source,node)
            start_args=call.index('.to_csv(')+len('.to_csv(')
            replacement='write_csv('+owner+', '+call[start_args:]
        elif node.func.attr=='write_text' and node.args:
            arg=node.args[0]
            if not (isinstance(arg,ast.BinOp) and isinstance(arg.left,ast.Call) and isinstance(arg.left.func,ast.Attribute) and arg.left.func.attr=='dumps'): continue
            parent_function=next((f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef) and f.name=='write_json'),None)
            if parent_function and parent_function.lineno<=node.lineno<=parent_function.end_lineno: continue
            replacement='write_json('+ast.get_source_segment(source,node.func.value)+', '+ast.get_source_segment(source,arg.left.args[0])+')'
        else: continue
        edits.append((position(node.lineno,node.col_offset),position(node.end_lineno,node.end_col_offset),replacement))
    for start,end,replacement in sorted(edits,reverse=True): source=source[:start]+replacement+source[end:]
    if path.name=='scan.py':
        source=source.replace('from sterile_fit.output import plot_three_plus_one_scan','from sterile_fit.output import write_csv, write_json, plot_three_plus_one_scan')
        source=source.replace('import matplotlib.pyplot as plt\n','')
    path.write_text(source,encoding='utf-8')


def move_callbacks():
    scan=ROOT/'src/sterile_fit/scan.py'
    source=scan.read_text(encoding='utf-8'); lines=source.splitlines(keepends=True)
    names={'_hypothesis_pairs','_objective_for_toy','extended_hypothesis_pairs','extended_objective_for_toy'}
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    destination=ROOT/'src/sterile_fit/adapter.py'
    with destination.open('a',encoding='utf-8') as stream:
        stream.write('\n\n# Hypothesis/observation adapters. Original arithmetic order is retained.\nfrom sterile_fit.core.calibration import GaussianHypothesis\n')
        for node in functions: stream.write('\n'+''.join(lines[node.lineno-1:node.end_lineno])+'\n')
    for node in sorted(functions,key=lambda n:n.lineno,reverse=True): lines[node.lineno-1:node.end_lineno]=[]
    source=''.join(lines).replace('from sterile_fit.adapter import build_three_plus_one_analysis','from sterile_fit.adapter import _hypothesis_pairs, _objective_for_toy, extended_hypothesis_pairs, extended_objective_for_toy, build_three_plus_one_analysis')
    scan.write_text(source,encoding='utf-8')
    for relative in ['studies/structure_migration/check_parity.py','studies/three_plus_one_toy_distribution_fit/run.py']:
        path=ROOT/relative; code=path.read_text(encoding='utf-8')
        for n in ast.parse(code).body:
            if isinstance(n,ast.ImportFrom) and n.module=='sterile_fit.scan':
                moved=[a for a in n.names if a.name in names]; kept=[a for a in n.names if a.name not in names]
                def imp(module,aliases): return 'from '+module+' import '+', '.join(a.name+(' as '+a.asname if a.asname else '') for a in aliases)+'\n' if aliases else ''
                rows=code.splitlines(keepends=True); rows[n.lineno-1:n.end_lineno]=[imp('sterile_fit.adapter',moved)+imp('sterile_fit.scan',kept)]; code=''.join(rows)
        path.write_text(code,encoding='utf-8')


def tidy_imports():
    # Hoist only ordinary imports, whose modules are already loaded unconditionally.
    # Keep ROOT aliases in their original position beside path declarations.
    for path in (ROOT/'src/sterile_fit').rglob('*.py'):
        source=path.read_text(encoding='utf-8'); lines=source.splitlines(keepends=True)
        nodes=ast.parse(source).body; imports=[]; removed=[]
        for node in nodes:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                if isinstance(node,ast.ImportFrom) and node.module=='__future__': continue
                if isinstance(node,ast.ImportFrom) and any(a.asname and a.asname.endswith('ROOT') for a in node.names): continue
                line=ast.get_source_segment(source,node)+'\n'
                if line not in imports: imports.append(line)
                removed.append(node)
        for node in reversed(removed): lines[node.lineno-1:node.end_lineno]=[]
        source=''.join(lines)
        if 'from __future__ import annotations\n' in source: source=source.replace('from __future__ import annotations\n','from __future__ import annotations\n\n'+''.join(imports),1)
        elif imports: source=''.join(imports)+source
        source=re.sub(r'\n{5,}','\n\n\n',source)
        path.write_text(source,encoding='utf-8')


if __name__=='__main__':
    route_writers(ROOT/'src/sterile_fit/output.py')
    route_writers(ROOT/'src/sterile_fit/scan.py')
    move_callbacks()
    tidy_imports()
