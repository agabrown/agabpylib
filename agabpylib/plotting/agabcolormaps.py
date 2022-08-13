"""
Custom colormaps:

run agabcolormaps.show_color_maps() to see what these colormaps look like.
"""

from matplotlib import rcParams, colors
from numpy import genfromtxt, linspace, empty, flipud, vstack, clip, where, floor
import matplotlib.pyplot as plt
import os

_LUTSIZE = rcParams["image.lut"]

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    return os.path.join(_ROOT, "data", path)


_Blackbody_data = genfromtxt(
    get_data("bbr_color.txt"),
    dtype=None,
    skip_header=19,
    comments="#",
    names=("temp", "cmf", "x", "y", "P", "r", "g", "b", "R", "G", "B", "hexRGB"),
    encoding=None,
)
npoints = int(len(_Blackbody_data["temp"]) / 2)
indices = where(_Blackbody_data["cmf"] == "10deg")
r_bb = _Blackbody_data["r"][indices]
g_bb = _Blackbody_data["g"][indices]
b_bb = _Blackbody_data["b"][indices]
_Planckian_temperatures = _Blackbody_data["temp"][indices] * 1.0
_Planckian_data = empty((npoints, 3))
_Planckian_data[:, 0] = r_bb
_Planckian_data[:, 1] = g_bb
_Planckian_data[:, 2] = b_bb


#
# Planckian locus in CIE space translated to RGB. Gives the colours of blackbodies in the range
# 1000 < T < 40000 K
def planckian_locus(tbb):
    """
    Interpolate the RGB values of the CIE colour on the Planckian locus in CIE space. Use the table
    http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html. Input temperatures
    outside the range [1000, 40000] K are clipped to that range.

    Parameters
    ----------

    tbb - Blackbody temperature in Kelvin

    Returns
    -------

    sRGB R, G, B values normalized to [0,1]
    """

    temp = clip(tbb, 1000.0, 40000.0)
    if temp == 1000.0:
        return tuple(_Planckian_data[0])
    elif temp == 40000.0:
        return tuple(_Planckian_data[-1])
    else:
        index = where(_Planckian_temperatures == floor(temp / 100.0) * 100.0)[0][0]
        amt = (temp - _Planckian_temperatures[index]) / (
            _Planckian_temperatures[index + 1] - _Planckian_temperatures[index]
        )
        rint = _Planckian_data[index, 0] + amt * (
            _Planckian_data[index + 1, 0] - _Planckian_data[index, 0]
        )
        gint = _Planckian_data[index, 1] + amt * (
            _Planckian_data[index + 1, 1] - _Planckian_data[index, 1]
        )
        bint = _Planckian_data[index, 2] + amt * (
            _Planckian_data[index + 1, 2] - _Planckian_data[index, 2]
        )
        return rint, gint, bint


# Varies from white to blue to yellow to red (IDV plots)
#
_WBYR_data = {
    "red": [(0.0, 1.0, 1.0), (0.3, 0.0, 0.0), (0.6, 1.0, 1.0), (1.0, 1.0, 1.0)],
    "green": [(0.0, 1.0, 1.0), (0.3, 0.0, 0.0), (0.6, 1.0, 1.0), (1.0, 0.0, 0.0)],
    "blue": [(0.0, 1.0, 1.0), (0.3, 1.0, 1.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0)],
}

# Varies from unsaturated blue to yellow to red.
#
_UBYR_data = {
    "red": [
        (0.0, 191.0 / 255.0, 191.0 / 255.0),
        (0.3, 0.0, 0.0),
        (0.6, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    "green": [
        (0.0, 191.0 / 255.0, 191.0 / 255.0),
        (0.3, 1.0, 1.0),
        (0.6, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    "blue": [(0.0, 1.0, 1.0), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0)],
}

# Varies from unsaturated to saturated blue to yellow to red (IDV plots)
#
_UBBYR_data = {
    "red": [
        (0.0, 191.0 / 255.0, 191.0 / 255.0),
        (0.3, 0.0, 0.0),
        (0.6, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    "green": [
        (0.0, 191.0 / 255.0, 191.0 / 255.0),
        (0.3, 0.0, 0.0),
        (0.6, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    "blue": [(0.0, 1.0, 1.0), (0.3, 1.0, 1.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0)],
}

# Berry Holl's POSNEG map.
#
_BerryPosNeg_data = {
    "red": [
        (0.0, 160.0 / 255.0, 160.0 / 255.0),
        (0.3, 0.0, 0.0),
        (0.475, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (0.525, 0.0, 0.0),
        (0.7, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    "green": [
        (0.0, 32.0 / 255.0, 32.0 / 255.0),
        (0.3, 0.0, 0.0),
        (0.475, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.525, 1.0, 1.0),
        (0.7, 200.0 / 255.0, 200.0 / 255.0),
        (1.0, 0.0, 0.0),
    ],
    "blue": [
        (0.0, 240.0 / 255.0, 240.0 / 255.0),
        (0.3, 1.0, 1.0),
        (0.475, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.525, 1.0, 1.0),
        (0.7, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

# Berry Holl's POS map.
#
_BerryPos_data = {
    "red": [(0.0, 1.0, 1.0), (0.15, 0.0, 0.0), (0.4, 1.0, 1.0), (1.0, 1.0, 1.0)],
    "green": [
        (0.0, 1.0, 1.0),
        (0.15, 1.0, 1.0),
        (0.4, 200.0 / 255.0, 200.0 / 255.0),
        (1.0, 0.0, 0.0),
    ],
    "blue": [(0.0, 1.0, 1.0), (0.15, 1.0, 1.0), (0.4, 0.0, 0.0), (1.0, 0.0, 0.0)],
}

# Using data from BlueDarkOrange18.rgb
#
bdoData = genfromtxt(
    get_data("BlueDarkOrange18.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_BlueDarkOrange18_data = empty((18, 3))
for i in range(18):
    _BlueDarkOrange18_data[i][0] = bdoData["r"][i] / 255.0
    _BlueDarkOrange18_data[i][1] = bdoData["g"][i] / 255.0
    _BlueDarkOrange18_data[i][2] = bdoData["b"][i] / 255.0

segments = linspace(0, 1, 9)
_BluesAgab_data = {
    "red": [
        (segments[i], _BlueDarkOrange18_data[i][0], _BlueDarkOrange18_data[i][0])
        for i in range(9)
    ],
    "green": [
        (segments[i], _BlueDarkOrange18_data[i][1], _BlueDarkOrange18_data[i][1])
        for i in range(9)
    ],
    "blue": [
        (segments[i], _BlueDarkOrange18_data[i][2], _BlueDarkOrange18_data[i][2])
        for i in range(9)
    ],
}

# Using data from BlueDarkOrange19.rgb
#
bdoData = genfromtxt(
    get_data("BlueDarkOrange19.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_BlueDarkOrange19_data = empty((19, 3))
for i in range(19):
    _BlueDarkOrange19_data[i][0] = bdoData["r"][i] / 255.0
    _BlueDarkOrange19_data[i][1] = bdoData["g"][i] / 255.0
    _BlueDarkOrange19_data[i][2] = bdoData["b"][i] / 255.0

segments = linspace(0, 1, 19)
_BlueDarkOrange_data = {
    "red": [
        (segments[i], _BlueDarkOrange19_data[i][0], _BlueDarkOrange19_data[i][0])
        for i in range(19)
    ],
    "green": [
        (segments[i], _BlueDarkOrange19_data[i][1], _BlueDarkOrange19_data[i][1])
        for i in range(19)
    ],
    "blue": [
        (segments[i], _BlueDarkOrange19_data[i][2], _BlueDarkOrange19_data[i][2])
        for i in range(19)
    ],
}

# Using data from BlueGreen14.rgb
#
bg14Data = genfromtxt(
    get_data("BlueGreen14.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_BlueGreen14_data = empty((14, 3))
for i in range(14):
    _BlueGreen14_data[i][0] = bg14Data["r"][i] / 255.0
    _BlueGreen14_data[i][1] = bg14Data["g"][i] / 255.0
    _BlueGreen14_data[i][2] = bg14Data["b"][i] / 255.0

# Using data from StepSeq25.rgb
#
ss25Data = genfromtxt(
    get_data("StepSeq25.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_StepSeq25_data = empty((25, 3))
for i in range(25):
    _StepSeq25_data[i][0] = ss25Data["r"][i] / 255.0
    _StepSeq25_data[i][1] = ss25Data["g"][i] / 255.0
    _StepSeq25_data[i][2] = ss25Data["b"][i] / 255.0

# Using data from CyanOrange14.rgb
#
co14Data = genfromtxt(
    get_data("CyanOrange14.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_CyanOrange14_data = empty((14, 3))
for i in range(14):
    _CyanOrange14_data[i][0] = co14Data["r"][i] / 255.0
    _CyanOrange14_data[i][1] = co14Data["g"][i] / 255.0
    _CyanOrange14_data[i][2] = co14Data["b"][i] / 255.0

# Using data from BlueDarkYellow18.rgb
#
bdyData = genfromtxt(
    get_data("BlueDarkYellow18.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_BlueDarkYellow18_data = empty((18, 3))
for i in range(18):
    _BlueDarkYellow18_data[i][0] = bdyData["r"][i] / 255.0
    _BlueDarkYellow18_data[i][1] = bdyData["g"][i] / 255.0
    _BlueDarkYellow18_data[i][2] = bdyData["b"][i] / 255.0

# Using data from BlueYellow14.rgb
#
by14Data = genfromtxt(
    get_data("BlueYellow14.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_BlueYellow14_data = empty((14, 3))
for i in range(14):
    _BlueYellow14_data[i][0] = by14Data["r"][i] / 255.0
    _BlueYellow14_data[i][1] = by14Data["g"][i] / 255.0
    _BlueYellow14_data[i][2] = by14Data["b"][i] / 255.0

# Using data from LightBlueToDarkBlue10.rgb
#
bu10Data = genfromtxt(
    get_data("LightBlueToDarkBlue10.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_LightBlueToDarkBlue10_data = empty((10, 3))
for i in range(10):
    _LightBlueToDarkBlue10_data[i][0] = bu10Data["r"][i] / 255.0
    _LightBlueToDarkBlue10_data[i][1] = bu10Data["g"][i] / 255.0
    _LightBlueToDarkBlue10_data[i][2] = bu10Data["b"][i] / 255.0

segments = linspace(0, 1, 10)
_LightBlueToDarkBlue_data = {
    "red": [
        (
            segments[i],
            _LightBlueToDarkBlue10_data[i][0],
            _LightBlueToDarkBlue10_data[i][0],
        )
        for i in range(10)
    ],
    "green": [
        (
            segments[i],
            _LightBlueToDarkBlue10_data[i][1],
            _LightBlueToDarkBlue10_data[i][1],
        )
        for i in range(10)
    ],
    "blue": [
        (
            segments[i],
            _LightBlueToDarkBlue10_data[i][2],
            _LightBlueToDarkBlue10_data[i][2],
        )
        for i in range(10)
    ],
}

# Using data from Cat_12.rgb
#
cat12data = genfromtxt(
    get_data("Cat_12.rgb"),
    dtype=None,
    skip_header=7,
    names=("r", "g", "b"),
    encoding=None,
)
_Cat12_data = empty((12, 3))
for i in range(12):
    _Cat12_data[i][0] = cat12data["r"][i] / 255.0
    _Cat12_data[i][1] = cat12data["g"][i] / 255.0
    _Cat12_data[i][2] = cat12data["b"][i] / 255.0

PlanckianLocus = colors.ListedColormap(_Planckian_data, name="PlanckianLocus")
WhiteBlueYellowRed = colors.LinearSegmentedColormap(
    "WhiteBlueYellowRed", _WBYR_data, _LUTSIZE
)
UnsBlueSatBlueYellowRed = colors.LinearSegmentedColormap(
    "UnsBlueSatBlueYellowRed", _UBBYR_data, _LUTSIZE
)
UnsBlueYellowRed = colors.LinearSegmentedColormap(
    "UnsBlueYellowRed", _UBYR_data, _LUTSIZE
)
BerryPosNeg = colors.LinearSegmentedColormap("BerryPosNeg", _BerryPosNeg_data, _LUTSIZE)
BerryPos = colors.LinearSegmentedColormap("BerryPos", _BerryPos_data, _LUTSIZE)
BlueDarkOrange18 = colors.ListedColormap(
    _BlueDarkOrange18_data, name="BlueDarkOrange18"
)
BlueDarkOrange = colors.LinearSegmentedColormap(
    "BlueDarkOrange", _BlueDarkOrange_data, _LUTSIZE
)
BluesAgab = colors.LinearSegmentedColormap("BluesAgab", _BluesAgab_data, _LUTSIZE)
BlueGreen14 = colors.ListedColormap(_BlueGreen14_data, name="BlueGreen14")
StepSeq25 = colors.ListedColormap(_StepSeq25_data, name="StepSeq25")
CyanOrange14 = colors.ListedColormap(_CyanOrange14_data, name="CyanOrange14")
BlueDarkYellow18 = colors.ListedColormap(
    _BlueDarkYellow18_data, name="BlueDarkYellow18"
)
BlueYellow14 = colors.ListedColormap(_BlueYellow14_data, name="BlueYellow14")
LightBlueToDarkBlue10 = colors.ListedColormap(
    _LightBlueToDarkBlue10_data, name="LightBlueToDarkBlue10"
)
LightBlueToDarkBlue = colors.LinearSegmentedColormap(
    "LightBlueToDarkBlue", _LightBlueToDarkBlue_data, _LUTSIZE
)
Cat12 = colors.ListedColormap(_Cat12_data, name="Cat12")

datad = {
    "PlanckianLocus": _Planckian_data,
    "WhiteBlueYellowRed": _WBYR_data,
    "UnsBlueSatBlueYellowRed": _UBBYR_data,
    "UnsBlueYellowRed": _UBYR_data,
    "BerryPosNeg": _BerryPosNeg_data,
    "BerryPos": _BerryPos_data,
    "BlueDarkOrange18": _BlueDarkOrange18_data,
    "BlueDarkOrange": _BlueDarkOrange_data,
    "BluesAgab": _BluesAgab_data,
    "BlueGreen14": _BlueGreen14_data,
    "StepSeq25": _StepSeq25_data,
    "CyanOrange14": _CyanOrange14_data,
    "BlueDarkYellow18": _BlueDarkYellow18_data,
    "BlueYellow14": _BlueYellow14_data,
    "LightBlueToDarkBlue10": _LightBlueToDarkBlue10_data,
    "LightBlueToDarkBlue": _LightBlueToDarkBlue_data,
    "Cat12": _Cat12_data,
}

cmapd = {
    "PlanckianLocus": PlanckianLocus,
    "WhiteBlueYellowRed": WhiteBlueYellowRed,
    "UnsBlueSatBlueYellowRed": UnsBlueSatBlueYellowRed,
    "UnsBlueYellowRed": UnsBlueYellowRed,
    "BerryPosNeg": BerryPosNeg,
    "BerryPos": BerryPos,
    "BlueDarkOrange18": BlueDarkOrange18,
    "BlueDarkOrange": BlueDarkOrange,
    "BluesAgab": BluesAgab,
    "BlueGreen14": BlueGreen14,
    "StepSeq25": StepSeq25,
    "CyanOrange14": CyanOrange14,
    "BlueDarkYellow18": BlueDarkYellow18,
    "BlueYellow14": BlueYellow14,
    "LightBlueToDarkBlue10": LightBlueToDarkBlue10,
    "LightBlueToDarkBlue": LightBlueToDarkBlue,
    "Cat12": Cat12,
}


# reverse all the colormaps.
# reversed colormaps have '_r' appended to the name.
def _revcmap(data):
    data_r = {}
    for key, val in data.items():
        val = list(val)
        valrev = val[::-1]
        valnew = []
        for a, b, c in valrev:
            valnew.append((1.0 - a, b, c))
        data_r[key] = valnew
    return data_r


#
# NOTE the construct in the next line is needed in python3 because just asking for datad.keys() will
# result in a runtime error "dictionary changed size during iteration".
_cmapnames = datad.copy().keys()
for _cmapname in _cmapnames:
    _cmapname_r = _cmapname + "_r"
    if isinstance(cmapd[_cmapname], colors.LinearSegmentedColormap):
        _cmapdat_r = _revcmap(datad[_cmapname])
        locals()[_cmapname_r] = colors.LinearSegmentedColormap(
            _cmapname_r, _cmapdat_r, _LUTSIZE
        )
    else:
        _cmapdat_r = flipud(datad[_cmapname])
        locals()[_cmapname_r] = colors.ListedColormap(_cmapdat_r, name=_cmapname_r)
    datad[_cmapname_r] = _cmapdat_r
    cmapd[_cmapname_r] = locals()[_cmapname_r]


def register_agab_maps():
    """
    Register the color maps defined above.
    """
    try:
        for m in cmapd.keys():
            plt.colormaps.register(cmapd[m], name=m)
    except ValueError as err:
        print(f"{err}\n\nExiting this loop as colour maps already registered.")


def show_color_maps():
    """
    Show all the color maps defined in the agabColorMaps module.
    """
    register_agab_maps()
    image = linspace(0, 1, _LUTSIZE).reshape(1, -1)
    image = vstack((image, image))

    # Get a list of the colormaps in this module.  Ignore the ones that end with
    # '_r' because these are simply reversed versions of ones that don't end
    # with '_r'
    maps = sorted(ma for ma in datad.keys() if not ma.endswith("_r"))
    nmaps = len(maps) + 1

    fig = plt.figure(figsize=(5, 10))
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    for j, ma in enumerate(maps):
        ax = plt.subplot(nmaps, 1, j + 1)
        plt.axis("off")
        plt.imshow(image, aspect="auto", cmap=plt.get_cmap(ma), origin="lower")
        pos = list(ax.get_position().bounds)
        fig.text(
            pos[0] + 0.5 * (pos[2] - pos[0]),
            pos[1],
            ma,
            fontsize=10,
            horizontalalignment="center",
        )

    plt.show()


if __name__ in "__main__":
    show_color_maps()
