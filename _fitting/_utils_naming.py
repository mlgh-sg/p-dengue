import re


def abbrev_surveillance(name):
    '''Abbreviate surveillance type and weighting from name.'''
    if name is None:
        return "nosurv"
    base = "surv"
    if "urban" in name:
        base = "urb_surv"
    weight = "p" if "pop_weighted" in name else "u"
    return f"{base}_{weight}"

def abbrev_urbanisation(name):
    '''Abbreviate urbanisation type and weighting from name.'''
    if name is None:
        return "nourb"
    base = "urb"
    weight = "p" if "pop_weighted" in name else "u"
    std = "_std" if "std" in name else ""
    return f"{base}_{weight}{std}"

def abbrev_stat(stat):
    '''Abbreviate statistic name.'''
    # remove spaces
    s = stat.replace(" ", "")
    
    # lag extraction: "(k)"
    lag = re.search(r"\((\d+)\)", s)
    lag_str = f"({lag.group(1)})" if lag else ""
    
    # check if _log is present
    has_log = ("_log" in s)
    
    # weighting
    if "pop_weighted" in s:
        w = "p"
    elif "unweighted" in s:
        w = "u"
    else:
        w = ""
    
    # remove weighting and lag, keep everything else
    base = re.sub(r"_?(pop_weighted|unweighted).*", "", s)
    
    # reattach _log if it was in original
    if has_log and not base.endswith("_log"):
        base += "_log"
    
    return f"{base}_{w}{lag_str}"