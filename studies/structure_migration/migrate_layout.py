"""One-use mechanical source regrouping. Preserves function text and comments.

Not an analysis dependency. The archived snapshot is read as text, never imported.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import importlib.util
import io
import json
from pathlib import Path
import re
import tokenize

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "frozen/before_four_block_layout_20260903"
DEST = "src/sterile_fit/"
GROUPS = {
    "core/three_plus_one.py": ["parameters.py", "models/base.py", "models/three_plus_one.py"],
    "core/one_plus_three_plus_one.py": ["one_plus_three_plus_one/parameters.py", "one_plus_three_plus_one/model.py"],
    "core/profile_three_plus_one.py": ["fitting.py"],
    "core/profile_one_plus_three_plus_one.py": ["one_plus_three_plus_one/fitting.py"],
    "core/likelihood.py": ["covariance.py", "likelihood.py"],
    "core/calibration.py": ["statistics/asymptotic_cls.py", "statistics/toy_cls.py"],
    "experiments/microboone/public_data.py": ["experiments/microboone/bnb/binning.py", "experiments/microboone/numi/binning.py", "experiments/microboone/bnb/published_inputs.py", "experiments/microboone/numi/published_inputs.py"],
    "experiments/microboone/response.py": ["experiments/microboone/bnb/archival_response.py", "experiments/microboone/bnb/adapters/archival_reco_60_to_bnb26.py"],
    "experiments/microboone/bnb.py": ["experiments/microboone/bnb/templates.py", "experiments/microboone/bnb/prediction.py", "experiments/microboone/bnb/workflow.py"],
    "experiments/microboone/numi.py": ["experiments/microboone/numi/event_prediction.py", "experiments/microboone/numi/prediction.py", "experiments/microboone/numi/workflow.py"],
    "experiments/microboone/joint.py": ["experiments/microboone/joint_bnb_numi.py"],
    "adapter.py": ["analysis/selection.py", "analysis/combination.py", "analysis/registry.py", "one_plus_three_plus_one/analysis.py"],
    "scan.py": [],
    "output.py": ["spectrum_plotting.py"],
}

# Script-owned names are prefixed before merging, avoiding shared-global changes.
SCRIPTS = [
    ("scripts/scan.py", "scan.py", "", "scan_three_plus_one"),
    ("scripts/one_plus_three_plus_one/scan.py", "scan.py", "extended", "scan_one_plus_three_plus_one"),
    ("scripts/check.py", "experiments/microboone/bnb.py", "check", "check_bnb"),
    ("scripts/experiments/microboone/bnb/build_anchor.py", "experiments/microboone/bnb.py", "anchor", "prepare_bnb_kernel"),
    ("scripts/experiments/microboone/bnb/prepare_bnb_total_covariance.py", "experiments/microboone/bnb.py", "covariance", "prepare_bnb_covariance"),
    ("scripts/experiments/microboone/bnb/normalize_archival_reco_matrices.py", "experiments/microboone/response.py", "normalize", "prepare_normalized_response"),
    ("scripts/experiments/microboone/bnb/adapt_archival_reco_to_bnb26.py", "experiments/microboone/response.py", "rebin", "prepare_bnb26_response"),
    ("scripts/experiments/microboone/numi/build_paper_weighted_flux.py", "experiments/microboone/numi.py", "flux", "prepare_numi_flux"),
    ("scripts/experiments/microboone/numi/build_four_channel_events.py", "experiments/microboone/numi.py", "events", "prepare_numi_kernel"),
    ("scripts/experiments/microboone/bnb/plot_published_bnb.py", "output.py", "bnb_plot", "plot_bnb_spectrum"),
    ("scripts/experiments/microboone/plot_bnb_numi_joint.py", "output.py", "joint_plot", "plot_joint_spectrum"),
    ("scripts/experiments/microboone/plot_figure1_nue_cc_fc.py", "output.py", "figure1", "plot_figure1_spectrum"),
    ("scripts/experiments/microboone/plot_public_inputs.py", "output.py", "inputs_plot", "plot_public_inputs"),
    ("scripts/experiments/microboone/numi/plot_flux_inputs.py", "output.py", "numi_plot", "plot_numi_flux_inputs"),
    ("studies/scan_result_comparison/plot_completed_scan_contours.py", "output.py", "comparison", "plot_completed_scan_contours"),
]


def module(path):
    path = path.removesuffix(".py").replace("/", ".")
    return path.removeprefix("src.")


MODULE_MAP = {module(DEST+old): module(DEST+new) for new, olds in GROUPS.items() for old in olds}
MODULE_MAP.update({module(old): module(DEST+new) for old,new,_,_ in SCRIPTS})
SYMBOL_MAP = {}


def text(path):
    base = SNAPSHOT if (SNAPSHOT/path).exists() else ROOT
    return (base/path).read_text(encoding="utf-8-sig")


def replace_ranges(source, changes):
    lines=source.splitlines(keepends=True)
    for start,end,value in sorted(changes,reverse=True): lines[start-1:end]=[value]
    return "".join(lines)


def rename_tokens(source, names):
    tokens=list(tokenize.generate_tokens(io.StringIO(source).readline))
    return tokenize.untokenize(t._replace(string=names.get(t.string,t.string)) if t.type==tokenize.NAME else t for t in tokens)


for old,new,prefix,main in SCRIPTS:
    tree=ast.parse(text(old)); rename={"main":main}
    if prefix:
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)): rename[node.name]=prefix+"_"+node.name.lstrip("_")
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                for target in (node.targets if isinstance(node,ast.Assign) else [node.target]):
                    if isinstance(target,ast.Name): rename[target.id]=prefix.upper()+"_"+target.id
        rename["main"]=main
    SYMBOL_MAP[module(old)]=rename
SYMBOL_MAP["sterile_fit.one_plus_three_plus_one.analysis"]={"_repository_path":"_extended_repository_path"}


def mapped_import(node, old_module, destination):
    if node.module=="__future__": return ""
    if node.level:
        source=importlib.util.resolve_name("."*node.level+(node.module or ""),old_module.rsplit(".",1)[0])
    else: source=node.module or ""
    # The former model package re-exported symbols from several destinations.
    if source=="sterile_fit.one_plus_three_plus_one":
        parts=defaultdict(list)
        for alias in node.names:
            name=alias.name
            target=("sterile_fit.adapter" if "Analysis" in name or "Experiment" in name or name.startswith("build_") else "sterile_fit.core.profile_one_plus_three_plus_one" if "Fit" in name or "Profile" in name or name.startswith(("profile_","prefit_")) else "sterile_fit.core.one_plus_three_plus_one")
            if target!=destination: parts[target].append(name+(" as "+alias.asname if alias.asname else ""))
        return "".join("from "+key+" import "+", ".join(values)+"\n" for key,values in parts.items())
    if source=="sterile_fit.statistics": source="sterile_fit.core.calibration"
    if source=="sterile_fit.analysis": source="sterile_fit.adapter"
    target=MODULE_MAP.get(source,source)
    if target==destination: return ""
    aliases=[]
    for alias in node.names:
        if alias.name.startswith("prefit_"): continue
        name=SYMBOL_MAP.get(source,{}).get(alias.name,alias.name)
        asname=alias.asname or (alias.name if name!=alias.name else None)
        aliases.append(name+(" as "+asname if asname else ""))
    return "from "+target+" import "+", ".join(aliases)+"\n" if aliases else ""


def convert(source, old_path, destination, rename=None):
    changes=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.ImportFrom): changes.append((node.lineno,node.end_lineno,mapped_import(node,module(old_path),destination)))
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            changes.append((node.lineno,node.end_lineno,"# "+node.value.value.splitlines()[0]+"\n"))
        elif isinstance(node,ast.If) and "__name__" in ast.unparse(node.test): changes.append((node.lineno,node.end_lineno,""))
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ROOT" for t in node.targets):
            changes.append((node.lineno,node.end_lineno,"from sterile_fit.paths import REPOSITORY_ROOT as ROOT\n"))
        elif isinstance(node,ast.FunctionDef) and node.name.startswith("prefit_"):
            changes.append((node.lineno,node.end_lineno,"# Standalone global prefit archived; constrained profile is retained.\n"))
    source=replace_ranges(source,changes)
    return rename_tokens(source,rename or {})


def profile_only(source):
    source=source.replace('choices=("prefit", "appearance-profile",', 'choices=("appearance-profile",')
    source=source.replace('        "prefit",\n','')
    source=re.sub(r'    global_best = min\(.*?\n    \)\n','',source,flags=re.S)
    start=source.index('    if arguments.mode == "prefit":\n')
    end=source.index('    elif arguments.mode == "appearance-profile":',start)
    source=source[:start]+source[end:].replace('    elif arguments.mode == "appearance-profile":','    if arguments.mode == "appearance-profile":',1)
    source=re.sub(r'            if point.best_fit.chi2 < global_best.chi2:\n                global_best = point.best_fit\n','',source)
    source=re.sub(r'    result_table\["delta_chi2"\] =.*?    null_parameters =', '    null_parameters =',source,flags=re.S)
    source=re.sub(r'        "best_fit_found": \{.*?\n        \},\n', '',source,flags=re.S)
    source=source.replace('            "prefit_seeds": [42, 137, 314],\n','')
    source=source.replace('"SciPy differential_evolution with polishing; full volume plus explicit s24=0 and s14=0 boundary profiles"','"unchanged coordinate-specific profile; no standalone global prefit"')
    source=source.replace('"lowest point found among the multiseed prefit and all evaluated profile points; not a proof of the mathematical global minimum"','"pointwise constrained profile only; no global-minimum claim"')
    # Historical conditions now have a known value: remove wrappers, not calculations.
    tree=ast.parse(source); changes=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If):
            expression=ast.unparse(node.test)
            if expression=="arguments.mode != 'prefit'":
                lines=source.splitlines(keepends=True)[node.lineno:node.end_lineno]
                changes.append((node.lineno,node.end_lineno,"".join(line[4:] if line.startswith('    ') else line for line in lines)))
            elif "arguments.mode == 'prefit'" in expression: changes.append((node.lineno,node.end_lineno,""))
    source=replace_ranges(source,changes)
    source=source.replace('values="delta_chi2"','values="chi2"')
    source=source.replace('/ "scans"','/ ("run1_non_toy" if arguments.cls_calibration == "analytic" else "run2_toy_mc")')
    return source


def no_spectrum_fit(source):
    changes=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and node.value.args and isinstance(node.value.args[0],ast.Constant) and node.value.args[0].value=="--compare-fit-points": changes.append((node.lineno,node.end_lineno,""))
        if isinstance(node,ast.If) and ast.unparse(node.test)=="arguments.compare_fit_points": changes.append((node.lineno,node.end_lineno,""))
    source=replace_ranges(source,changes)
    source=source.replace("arguments.compare_fit_points or arguments.compare_paper_figure1_points","arguments.compare_paper_figure1_points")
    source=source.replace("arguments.compare_fit_points","False")
    return source


def main():
    if (ROOT/DEST/"core/three_plus_one.py").exists(): raise RuntimeError("Migration already applied")
    assembled={name:[] for name in GROUPS}
    manifest=[]
    for destination,sources in GROUPS.items():
        for old in sources:
            path=DEST+old
            assembled[destination].append(convert(text(path),path,module(DEST+destination),SYMBOL_MAP.get(module(path))))
            manifest.append({"old":path,"new":DEST+destination})
    for path,destination,_,_ in SCRIPTS:
        source=text(path)
        if path=="scripts/scan.py": source=profile_only(source)
        if path.endswith("plot_published_bnb.py"): source=no_spectrum_fit(source)
        # Move scan visualization functions into the output block, unchanged.
        if path=="scripts/scan.py":
            start=source.index('    x_name = {',source.index('result_table.to_csv'))
            end=source.index('    metadata = {',start)
            plot=source[start:end]
            assembled["output.py"].append('def plot_three_plus_one_scan(result_table, output_directory, arguments, analysis):\n'+plot)
            source=source[:start]+'    plot_three_plus_one_scan(result_table, output_directory, arguments, analysis)\n'+source[end:]
            source='from sterile_fit.output import plot_three_plus_one_scan\n'+source
        if path=="scripts/one_plus_three_plus_one/scan.py":
            node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="_plot_result")
            function="".join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
            assembled["output.py"].append(function.replace('def _plot_result(','def plot_one_plus_three_plus_one_scan(').replace('_log_cell_edges(', '_extended_plot_log_cell_edges(')+'\n')
            edge=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="_log_cell_edges")
            assembled["output.py"].append("".join(source.splitlines(keepends=True)[edge.lineno-1:edge.end_lineno]).replace('def _log_cell_edges(','def _extended_plot_log_cell_edges(')+'\n')
            source=replace_ranges(source,[(node.lineno,node.end_lineno,'from sterile_fit.output import plot_one_plus_three_plus_one_scan as _plot_result\n')])
        assembled[destination].append(convert(source,path,module(DEST+destination),SYMBOL_MAP[module(path)]))
        manifest.append({"old":path,"new":DEST+destination})
    # 3+1 plot helper copied from scan, preserving validation and arithmetic.
    old_scan=text('scripts/scan.py'); n=next(n for n in ast.parse(old_scan).body if isinstance(n,ast.FunctionDef) and n.name=='_log_cell_edges')
    assembled['output.py'].insert(0,''.join(old_scan.splitlines(keepends=True)[n.lineno-1:n.end_lineno])+'\n')
    for destination,blocks in assembled.items():
        header='"""'+destination+': regrouped existing implementations; see docs/ARCHITECTURE.md."""\nfrom __future__ import annotations\n'
        if destination=='output.py': header+='from matplotlib.lines import Line2D\n'
        source=header+'\n\n'.join(blocks)+'\n'
        ast.parse(source)
        path=ROOT/DEST/destination; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(source,encoding='utf-8')
    # All remaining live consumers follow the new import paths; no compatibility shims.
    for folder in ['tests','studies']:
        for path in (ROOT/folder).rglob('*.py'):
            if path==Path(__file__).resolve() or '/data/' in path.as_posix(): continue
            source=path.read_text(encoding='utf-8-sig'); changes=[]
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.ImportFrom) and not node.level:
                    new=mapped_import(node,module(path.relative_to(ROOT).as_posix()),'__external__')
                    if new and node.module!='__future__':
                        indent=' '*(node.col_offset)
                        changes.append((node.lineno,node.end_lineno,indent+new))
            if changes: path.write_text(replace_ranges(source,changes),encoding='utf-8')
    # Verify every retired live file was copied byte-for-byte before removing it.
    retired=[ROOT/DEST/old for olds in GROUPS.values() for old in olds]
    retired += [p for p in (ROOT/'src/sterile_fit').rglob('__init__.py') if '/core/' not in p.as_posix()]
    retired += list((ROOT/'scripts').rglob('*.py'))
    for path in dict.fromkeys(retired):
        relative=path.relative_to(ROOT)
        if not (SNAPSHOT/relative).is_file() or path.read_bytes()!=(SNAPSHOT/relative).read_bytes(): raise RuntimeError('Archive mismatch: '+str(relative))
        path.unlink()
    for relative in ['src/sterile_fit/__init__.py','src/sterile_fit/core/__init__.py','src/sterile_fit/experiments/__init__.py','src/sterile_fit/experiments/microboone/__init__.py']:
        (ROOT/relative).write_text('"""Package marker. Import concrete modules explicitly."""\n',encoding='utf-8')
    # This historical syntax-only flux extractor is a provenance tool, not daily preparation.
    flux_tool=ROOT/'studies/bnb_flux_provenance/prepare_visible_bnb_flux.py'
    flux_tool.parent.mkdir(parents=True,exist_ok=True)
    flux_tool.write_bytes((SNAPSHOT/'scripts/experiments/microboone/bnb/prepare_visible_bnb_flux.py').read_bytes())
    gui=ROOT/'ksquare/chi_square_gui.py'
    target=ROOT/'studies/chi_square_gui/chi_square_gui.py'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(gui.read_bytes()); gui.unlink()
    (ROOT/'studies/structure_migration/module_map.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Merged',len(manifest),'source sections into',len(assembled),'modules; original code retained in',SNAPSHOT)


if __name__=='__main__': main()
