"""Portfolio of ML demos (tabbed Gradio app).

Combines four projects into one Space:
  1. California Housing - regression
  2. Pima Diabetes - classification
  3. Online Retail - sequence model (segment + attention)
  4. Amazon Review Sentiment - text classification

Each tab lives in its own module (tab_*.py) and exposes a build_*_tab()
function returning a gr.Interface. Artifacts are loaded at import time inside
those modules.

Run locally:
    python assignments/portfolio_app/app.py
"""

import gradio as gr

from tab_housing import build_housing_tab
from tab_diabetes import build_diabetes_tab
from tab_retail import build_retail_tab
from tab_sentiment import build_sentiment_tab


def build_app() -> gr.TabbedInterface:
    tabs = [
        build_housing_tab(),
        build_diabetes_tab(),
        build_retail_tab(),
        build_sentiment_tab(),
    ]
    titles = ["Housing", "Diabetes", "Retail", "Sentiment"]
    return gr.TabbedInterface(tabs, titles, title="Peter Mangoro - ML Demos")


demo = build_app()


if __name__ == "__main__":
    demo.launch()
