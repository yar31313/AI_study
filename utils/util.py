import importlib.machinery
import importlib.util

def module_loader(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    loadermodule = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(loadermodule)
    return loadermodule