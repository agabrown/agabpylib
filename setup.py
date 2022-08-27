from setuptools import find_packages, setup

setup(
    name="agabpylib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "": ["LICENSE", "AUTHORS.md", "HISTORY.md", "INSTALL.md", "MANIFEST.in"],
        "agabpylib.plotting": ["data/*"],
        "agabpylib.simulation": ["data/*"],
        "agabpylib.gaia": ["data/*"],
    },
    include_package_data=True,
)
