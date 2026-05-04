# SRtools Web App

This directory contains the Streamlit app for interactive SR model exploration.
It is intentionally separate from the core `SRtools` package: the app imports
`SRtools` like any external user, and the core package does not depend on
Streamlit.

## Production install

```bash
python -m pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Local core development

Use this when developing the scientific package and testing the app against
your local checkout:

```bash
python -m pip install -e .
python -m pip install streamlit plotly
streamlit run app/streamlit_app.py
```

## Testing the app against a Git branch or tag

Replace the package line in a deployment-specific requirements file with a Git
URL, for example:

```text
srtools-aging @ git+https://github.com/navehr/SRtools.git@v0.1.0
```

Uploaded files are used in memory for the active Streamlit session only.
