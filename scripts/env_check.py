import sys, importlib
mods = ["pandas","numpy","statsmodels","pmdarima","matplotlib","openpyxl"]
print("Python:", sys.version)
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m}: {getattr(mod,'__version__','OK')}")
    except Exception as e:
        print(f"{m}: MISSING -> {e}")
