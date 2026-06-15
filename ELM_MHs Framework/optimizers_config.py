"""
Dictionary of metaheuristic optimizers to use with MhaElmClassifier.

Names like 'BaseGA', 'OriginalPSO', 'OriginalWOA' must be supported by IntelELM /
MEALPY. You can inspect them via:

    from intelelm import MhaElmClassifier
    print(MhaElmClassifier.SUPPORTED_OPTIMIZERS)
"""

ALL_OPTIMIZERS = {
    "GA": {
        "optim": "BaseGA",
        "optim_params": {"name": "GA", "epoch": 100, "pop_size": 30},
    },
    "PSO": {
        "optim": "OriginalPSO",
        "optim_params": {"name": "PSO", "epoch": 100, "pop_size": 30},
    },
    "WOA": {
        "optim": "OriginalWOA",
        "optim_params": {"name": "WOA", "epoch": 100, "pop_size": 30},
    },
    "GWO": {
        "optim": "OriginalGWO",
        "optim_params": {"name": "GWO", "epoch": 100, "pop_size": 30},
    },
    "HGS": {
        "optim": "OriginalHGS",
        "optim_params": {"name": "HGS", "epoch": 100, "pop_size": 30},
    },
     "HHO": {
        "optim": "OriginalHHO",
        "optim_params": {"name": "HHO", "epoch": 100, "pop_size": 30},
    },
    "MFO": {
        "optim": "OriginalMFO",
        "optim_params": {"name": "MFO", "epoch": 100, "pop_size": 30},
    },
     "MGO": {
        "optim": "OriginalMGO",
        "optim_params": {"name": "MGO", "epoch": 100, "pop_size": 30},
    },
        "DSGWO": {
        "optim": "DS_GWO",          # Mealpy class: GWO.DS_GWO
        "optim_params": {
            "name": "DS_GWO",
            "epoch": 100,          # tune as needed
            "pop_size": 30,        # tune as needed
            "explore_ratio": 0.4,  # DS-GWO-specific
            "n_groups": 5,         # DS-GWO-specific
        },
    },
       "ERGWO": {
        "optim": "ER_GWO",        # Efficient and Robust GWO
        "optim_params": {
            "name": "ER_GWO",
            "epoch": 100,         # tune later if needed
            "pop_size": 30,       # tune later
            # add ER-GWO-specific hyper-params here if Mealpy exposes them
        },
    },
        "GWOWOA": {                         # NEW: Hybrid GWO–WOA
        "optim": "GWO_WOA",            # <- confirm this key in SUPPORTED_OPTIMIZERS
        "optim_params": {
            "name": "GWO_WOA",
            "epoch": 100,
            "pop_size": 30,
            # add any hybrid-specific params here if Mealpy exposes them
        },
    },
    "IOBLGWO": {                     # NEW: Improved OBL-GWO
        "optim": "IOBL_GWO",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "IOBL_GWO",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "HPSOTVAC": {                     # NEW: Improved OBL-GWO
        "optim": "HPSO_TVAC",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "HPSO_TVAC",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "DGSKA": {                     # NEW: Improved OBL-GWO
        "optim": "DevGSKA",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "DevGSKA",
            "epoch": 100,
            "pop_size": 30,
            "pb": 0.1, 
            "kr": 0.9,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "RUN": {                     # NEW: Improved OBL-GWO
        "optim": "OriginalRUN",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "OriginalRUN",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "SHADE": {                     # NEW: Improved OBL-GWO
        "optim": "L_SHADE",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "L_SHADE",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "MEO": {                     # NEW: Improved OBL-GWO
        "optim": "ModifiedEO",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "ModifiedEO",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "CLPSO": {                     # NEW: Improved OBL-GWO
        "optim": "CL_PSO",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "CL_PSO",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "HIWOA": {                     # NEW: Improved OBL-GWO
        "optim": "HI_WOA",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "HI_WOA",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "IncGWO": {                     # NEW: Improved OBL-GWO
        "optim": "IncrementalGWO",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "IncrementalGWO",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "SeaHO": {                     # NEW: Improved OBL-GWO
        "optim": "OriginalSeaHO",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "OriginalSeaHO",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    "MRFO": {                     # NEW: Improved OBL-GWO
        "optim": "OriginalMRFO",        # <- replace with the exact key you saw
        "optim_params": {
            "name": "OriginalMRFO",
            "epoch": 100,
            "pop_size": 30,
            # add IOBL-GWO specific params here if Mealpy exposes them
        },
    },
    # Add more MHAs as needed, following the same pattern.
}

# Control which optimizers actually run:
# - Set to None or [] to run ALL optimizers in ALL_OPTIMIZERS.
# - Or set to a list of keys, e.g. ["GA", "WOA"].

#ACTIVE_OPTIMIZERS = None   # or ["GA", "WOA"]
# The 11 metaheuristics reported in the paper:
ACTIVE_OPTIMIZERS = ["GA", "RUN", "MEO", "CLPSO", "HIWOA", "GWO", "HGS", "HHO", "SeaHO", "MGO", "GWOWOA"]


def get_selected_optimizers():
    """
    Return a dict of optimizers that should be used in experiments.
    """
    if not ACTIVE_OPTIMIZERS:  # None or empty => use all
        return ALL_OPTIMIZERS

    selected = {}
    for name in ACTIVE_OPTIMIZERS:
        if name in ALL_OPTIMIZERS:
            selected[name] = ALL_OPTIMIZERS[name]
    return selected
