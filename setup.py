import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

long_description = (HERE / "README.md").read_text(encoding="utf8")

setup(
	long_description=long_description,
	long_description_content_type="text/markdown",
)
