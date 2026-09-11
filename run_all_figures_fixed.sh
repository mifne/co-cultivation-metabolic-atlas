#!/bin/bash
set -e

# Run all data extractions and plotting for the paper figures
echo "Extracting Fig 2..."
python paper_figures/extract_fig2_data.py
python paper_figures/plot_fig2.py

echo "Extracting Fig 3 (1-min dt, 72h)..."
python paper_figures/extract_fig3_data.py
python paper_figures/plot_fig3.py

echo "Extracting Fig 4/5 (240h)..."
python paper_figures/extract_fig4_5_data.py
python paper_figures/plot_fig4.py
python paper_figures/plot_fig5_revised.py

echo "Extracting Fig 5 Stochastic (3 episodes)..."
python paper_figures/extract_fig5_stochastic.py

echo "Extracting Fig 7 Shock Test (168h)..."
python paper_figures/extract_fig7_data.py

echo "Plotting Fig 6 and Fig 7..."
python paper_figures/plot_fig6_7.py

echo "Extracting Fig 8/10 (XAI & Pacing)..."
python paper_figures/extract_fig8_xai.py
python paper_figures/plot_fig8_10.py

echo "Extracting Fig 9 (Flux Log)..."
python paper_figures/extract_fig9_flux.py
python paper_figures/plot_fig9.py

echo "All figures generated successfully! Check paper_figures/ directory."
