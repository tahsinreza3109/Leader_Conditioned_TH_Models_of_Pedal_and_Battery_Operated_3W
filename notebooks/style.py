"""House style shared by all notebooks: one colour scheme, Cambria, 600-dpi PNG, Cambria Excel.

The same colour always means the same thing across the manuscript:
  PR (pedal rickshaw)            green      BTW (battery three-wheeler)      orange
  leader classes                 4W slate, BTW orange, CNG mustard, MT_2W purple, NMT green (NMT includes PR)
  YOLO vehicle classes           matched to the leader classes (3W orange, CNG mustard, 4W slate, Bike purple, Bicycle green)
  distribution families          Weibull (the chosen family) blue; others distinct; exponential grey
  policy scenarios               no limit = ink, 25 km/h = blue, 15 km/h = crimson
  outcomes under a limit         not affected = light grey, follows at the limit = blue, following breaks up = crimson
"""
from pathlib import Path
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font

FONT = "Cambria"
INK, MUTED, LIGHT = "#2d3142", "#4f5d75", "#c9ccd5"
BLUE, CRIMSON = "#2b7bba", "#b2182b"

PAL = {
    "ink": INK, "muted": MUTED, "light": LIGHT,
    "target": {"PR": "#1b9e77", "BTW": "#d95f02"},
    "target_sub": {"PR": "#1b9e77", "BTW": "#d95f02", "Mod_BTW": "#fdae6b"},
    "leader": {"4W": MUTED, "BTW": "#d95f02", "CNG": "#e6ab02", "MT_2W": "#7570b3", "NMT": "#1b9e77"},
    "yolo_class": {"3W": "#d95f02", "CNG": "#e6ab02", "4W": MUTED, "Bus": "#8c510a", "Bike": "#7570b3", "Bicycle": "#1b9e77"},
    "family": {"lognormal": "#a6761d", "loglogistic": "#e7298a", "weibull": "#2166ac", "gamma": "#66a61e",
               "gengamma": INK, "exponential": "0.6"},
    "variable": {"Off_Centered": "#8c510a", "Occupied": "#c51b7d", "Leader_speed_5kmh": "#35978f"},
    "component": {"logH": INK, "logS": "#8073ac", "neg_logv": "#e08214"},
    "cap": {"observed": INK, 25: BLUE, 15: CRIMSON},
    "outcome": {1: LIGHT, 2: CRIMSON, 3: BLUE},
    "exposure": {"still": INK, "lifted_speed": BLUE, "removed": CRIMSON},
    "lane": {"mixed_observed": INK, "mixed_cap25": "#8c8fa3", "lane_A": BLUE, "lane_B": "#5aae61"},
    "training": {"mAP50": "#2166ac", "mAP50_95": CRIMSON},
}


def use_style():
    plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": FONT,
                         "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold", "font.size": 9})


def save_fig(fig, folder, name):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.png"; p.unlink(missing_ok=True)
    fig.savefig(p, dpi=600, bbox_inches="tight"); print("saved", p.name)


def cambria_xlsx(path):
    wb = load_workbook(path)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name=FONT, bold=(c.row == 1), size=11)
    wb.save(path)


use_style()
