# viz/factory/state_line.py
from .registry import register
from ..base import Plotter
from ...analysis import build_variable

@register("state_line")
class StateLinePlotter(Plotter):
    def __init__(self, config: dict):
        self.type = "state_line"
        self.config = config
        self.varname = config.get("varname")
        self.var_cfg = dict(config.get("var_cfg", {}))
        if not self.varname:
            raise ValueError("StateLinePlotter requires 'varname' in config.")
        # normalize synonyms
        if "wvls" in self.var_cfg and "wvl_grid" not in self.var_cfg:
            self.var_cfg["wvl_grid"] = self.var_cfg.pop("wvls")

    def _fmt_label(self, long_label, units):
        units = (units or "").strip()
        return f"{long_label} [{units}]" if units else long_label
    
    def prep(self, population):
        #yvar = build_variable(self.varname, self.var_cfg)
        yvar = build_variable(name=self.varname, scope="population", var_cfg=self.var_cfg)
        # choose x-axis variable (simplified)
        if self.varname in ("Nccn", "frac_ccn"):
            xvar = build_variable("s_grid", self.var_cfg)
        elif self.varname in ("b_abs","b_scat","b_ext"):
            has_w = len(self.var_cfg.get("wvl_grid", [])) > 1
            has_rh = len(self.var_cfg.get("rh_grid", [])) > 1
            if has_w and has_rh:
                raise ValueError("state_line needs one varying axis (wavelength or RH).")
            elif has_w:
                xvar = build_variable(name="wvl_grid", scope="population", var_cfg=self.var_cfg)
            elif has_rh:
                xvar = build_variable(name="rh_grid", scope="population", var_cfg=self.var_cfg)
            else:
                raise ValueError(f"Variable {self.varname} has single wavelength and single RH value; cannot plot state line.")
            #xvar = build_variable("wvl_grid" if has_w else "rh_grid", "population", self.var_cfg)
        elif self.varname == "dNdlnD":
            xvar = build_variable("diam_grid","population",  self.var_cfg)
        else:
            raise ValueError(f"State line does not support '{self.varname}'.")

        x = xvar.compute(population)
        y = yvar.compute(population)
        if len(y) == 1:
            y = y[0]  # flatten single-value arrays
        if len(x) == 1:
            x = x[0]  # flatten single-value arrays
        print(x)
        print(y)
        print(self.var_cfg)
        if x is not None and len(x) != len(y):
            raise ValueError(f"x and y must be same length, got {len(x)} vs {len(y)}.")

        return {
            "x": x, "y": y,
            "xlabel": self._fmt_label(xvar.meta.long_label, getattr(xvar.meta, "units", "")),
            "ylabel": self._fmt_label(yvar.meta.long_label, getattr(yvar.meta, "units", "")),
            "xscale": "linear", "yscale": "linear",
        }

    def plot(self, population, ax, **kwargs):
        pd = self.prep(population)
        style = {**self.config.get("style", {}), **kwargs}
        print(pd['x'], pd['y'])
        if pd["x"] is None:
            ax.plot(pd["y"], **style)
        else:
            ax.plot(pd["x"], pd["y"])#, **style)
        ax.set_xlabel(pd["xlabel"]); ax.set_ylabel(pd["ylabel"])
        ax.set_xscale(pd["xscale"]); ax.set_yscale(pd["yscale"])
        return ax

def build(cfg):
    return StateLinePlotter(cfg)


# # viz/factory/state_line.py

# from .registry import register
# from ..base import Plotter
# from ...analysis import build_variable
# import numpy as np

# @register("state_line")
# class StateLinePlotter(Plotter):
#     def __init__(self, config):
#         self.type = "state_line"
#         self.config = config
#         var_cfgs = config.get("variables",{})
#         if not var_cfgs or len(var_cfgs) != 1:
#             raise ValueError("StateLinePlotter requires exactly one variable in 'variables' list.")
#         self.varname = varnames[0]
#         self.var_cfg = config.get("var_cfg", {})
#         if not self.varname:
#             raise ValueError("StateLinePlotter requires 'varname' in config.")
    
#     def prep(self, population):
#         # todo: validate var_cfg for the varname
        
#         yvar = build_variable(self.varname, self.var_cfg)  # ensure variable is built/registered
        
#         # todo: should this be moved to variable constrction?
#         if self.varname in ['Nccn', 'frac_ccn']:
#             xvar = build_variable("s_grid", self.var_cfg)  # ensure variable is built/registered
#         elif self.varname in ['b_abs','b_scat','b_ext']:
#             if len(self.var_cfg.get('wvls',[])) > 1 and len(self.var_cfg.get('rh_grid',[])) > 1:
#                 raise
#             elif len(self.var_cfg.get('wvls',[])) > 1:
#                 xvar = build_variable("wvl_grid", self.var_cfg)  # ensure variable is built/registered
#             elif len(self.var_cfg.get('rh_grid',[])) > 1:
#                 xvar = build_variable("rh_grid", self.var_cfg)  # ensure variable is built/registered
#             else:
#                 raise ValueError(f"Variable {self.varname} has single wavelength and single RH value; cannot plot state line.")
#         # todo: generalize density distributions?
#         elif self.varname in ['dNdlnD']:
#             xvar = build_variable("diam_grid", self.var_cfg)  # ensure variable is built/registered
#         else:
#             raise ValueError(f"State line plotter does not support variable '{self.varname}'")
        
#         if len(xvar.meta.unit) > 0:
#             xlabel = xvar.meta.long_label + " [" + f" ({xvar.meta.unit})" + "]"
#         else:
#             xlabel = xvar.meta.long_label

#         if len(yvar.meta.units) > 0:
#             ylabel = yvar.meta.long_label + " [" + f" ({yvar.meta.units})" + "]"
#         else:
#             ylabel = yvar.meta.long_label
        
#         plotdat = { 
#             'x': xvar.compute(population),
#             'y': yvar.compute(population),
#             'xlabel': xlabel,
#             'ylabel': ylabel,
#             'xscale': 'linear',
#             'yscale': 'linear'
#         }
#         return plotdat
    
#     def plot(self, population, ax, **kwargs):
#         plotdat = self.prep(population)
#         ax.plot(plotdat['x'], plotdat['y'], **kwargs)
#         ax.set_xlabel(plotdat['xlabel'])
#         ax.set_ylabel(plotdat['ylabel'])
#         ax.set_xscale(plotdat['xscale'])
#         ax.set_yscale(plotdat['yscale'])
#         return ax
