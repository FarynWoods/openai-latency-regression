# What Latency Can Tell Us About Emissions in Closed-Source LLMs: *A Partial Model and a Call for Transparency*

**The main contribution here is the paper, `report.ipynb`. Start there!**


## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install Poetry if you haven't
pip install poetry

# Clone the repo
git clone https://github.com/FarynWoods/openai-latency-regression
cd openai-latency-regression

# Install dependencies
poetry install
```

## License
See the [LICENSE](LICENSE.md) file for license rights and limitations (GNU GPLv3).


### References
References markdown generated from references.bib according to instructions in ref.md.

```
pandoc -t markdown_strict --citeproc ref.md -o references.md
```