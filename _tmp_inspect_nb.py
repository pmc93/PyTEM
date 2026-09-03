import json
path = r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\pyTEM\notebooks\8. pytem_schematic_figure.ipynb"
nb = json.load(open(path, encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    print(f"--- cell {i} ({cell['cell_type']}) ---")
    print(repr(src[:80]))
print("TOTAL CELLS", len(nb["cells"]))
