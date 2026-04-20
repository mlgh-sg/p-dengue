def color_switch(val, threshold,
                 color_below="background-color: #c6efce", color_above="background-color: #ffc7ce",
                 invert=False):
    '''Styling for dataframe: Color code for values above/below threshold.'''
    if isinstance(val, (int, float)):
        if (val < threshold) ^ invert:
            color = color_below
        else:
            color = color_above
    else:
        color = ""
    return color

def color_rhat(val):
    return color_switch(val, threshold=1.01)

def color_ess(x, n_draws):
    if isinstance(x, (int, float)):
        if x < n_draws / 5:
            return "background-color: red;"
        elif x < n_draws / 4:
            return "background-color: yellow;"
        else:
            return "background-color: lightgreen;"
    return ""