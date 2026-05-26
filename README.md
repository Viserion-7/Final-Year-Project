# A Neural Framework for Sandhi Splitting in Low-Resource Ayurvedic Sanskrit Texts

## Abstract

This project presents a neural framework for automatic Sandhi splitting in Sanskrit texts, specifically targeting low-resource Ayurvedic documents. Sandhi refers to the phonological processes that occur at word boundaries in Sanskrit, making it challenging to segment and process ancient texts. Our hybrid approach combines multiple neural architectures to effectively handle this complex linguistic problem, enabling better text processing and information extraction from classical Sanskrit corpora.

## Team & Guidance

**Guide:** Jayashree Nair

**Team Members:**
- Harigovind C B (AM.EN.U4AIE22119)
- Abhiram A S (AM.EN.U4AIE22002)
- Karthik Narayan C (AM.EN.U4AIE22028)

---

## Project Overview

Sandhi, a fundamental aspect of Sanskrit grammar, represents sound changes that occur at morpheme and word boundaries. In written Sanskrit texts, especially classical works like Ayurvedic manuscripts, these Sandhi transformations make it difficult to identify word boundaries and parse the text accurately. This project develops a neural framework to automatically reverse Sandhi transformations and segment Sanskrit text into individual words.

The project progresses through systematic phases, combining machine learning techniques with linguistic domain knowledge to achieve robust performance on low-resource datasets.

---

## Architecture Overview

### Hybrid Model Approach

Our framework employs a **hybrid neural architecture** that combines the strengths of multiple model types:
<p align="center">
  <span style="background:white; padding:12px; display:inline-block;">
    <img src="Final pipeline.png" width="100%">
  </span>
</p>

The pipeline integrates:
- **Encoder-Decoder Models** for sequence-to-sequence transformation
- **Attention Mechanisms** for focusing on relevant context
- **Transfer Learning** leveraging pre-trained language models
- **Domain-Specific Optimization** for Ayurvedic Sanskrit texts

For detailed architectural decisions and technical implementation, refer to the [Final Project Report](Final_Project_Report.pdf).

---

## Code Structure

The complete implementation is organized as follows:

### Main Implementation: `Final-Hybrid/`

The hybrid model implementation is located in the [`Final-Hybrid/`](Final-Hybrid/) directory:

- **[`final-hybrid.ipynb`](Final-Hybrid/final-hybrid.ipynb)** — Main hybrid model notebook
  - Model architecture definition and training pipeline
  - Data preparation and preprocessing steps
  - Training loops and optimization procedures

- **[`inference.ipynb`](Final-Hybrid/inference.ipynb)** — Inference and evaluation notebook
  - Model evaluation on test datasets
  - Inference examples and demonstrations
  - Performance metrics and analysis

- **[`app.py`](Final-Hybrid/app.py)** — Gradio web interface
  - Interactive web application for sandhi splitting
  - User-friendly interface for testing the model
  - Real-time prediction demonstrations

- **`byt5_sandhi_model/`** — Trained model artifacts
- **`data/`** — Preprocessed datasets used in training and evaluation

### Project Phases

The development process is documented in phase-specific folders:

- **[`Phase_1/`](Phase_1/)** — Initial exploration and baseline models
  - Data collection and analysis
  - Baseline model development
  - Feasibility studies

- **[`Phase_2/`](Phase_2/)** — Advanced modeling and refinement
  - Hybrid model experimentation
  - Performance optimization
  - Final model selection and tuning

### Supporting Materials

- **[`Report/`](Report/)** — AI and Plagarism Report for the Final Report.
- **[`Archives/`](Archives/)** — Historical versions and experimental runs

---

## Key Features

**Low-Resource Learning** — Effective performance on limited training data
**Hybrid Architecture** — Combines multiple neural approaches for robustness
**Sanskrit-Specific** — Domain-optimized for Ayurvedic texts
**Transfer Learning** — Leverages pre-trained models for improved accuracy
**Attention Mechanisms** — Contextual understanding of sandhi patterns

---

## References & Documentation

### Project Reports
- **[Final Project Report](Final_Project_Report.pdf)** — Comprehensive technical report with detailed methodology, experiments, and results.
- **[Final Review Presentation](Final%20Review%20PPT.pdf)** — Project presentation slides with key findings and visualizations.

### Repository Contents

```
0_FYP/
├── README.md                    (This file)
├── Final-Hybrid/                (Main hybrid implementation)
│   ├── final-hybrid.ipynb       (Model training)
│   ├── inference.ipynb          (Evaluation & inference)
│   ├── app.py                   (Web interface)
│   └── byt5_sandhi_model/       (Model artifacts)
├── Phase_1/                     (Initial exploration)
├── Phase_2/                     (Advanced development)
├── Report/                      (Reports)
├── Final_Project_Report.pdf     (Full technical report)
├── Final Review PPT.pdf         (Presentation)
├── Final pipeline.png           (Architecture diagram)
└── Archives/                    (Historical versions)
```

---

## Getting Started

### Exploring the Project

1. **Quick Overview** → Start with this README and view the [architecture diagram](#architecture-overview)
2. **Detailed Report** → Read the [Final Project Report](Final_Project_Report.pdf) for technical depth
3. **Implementation Details** → Browse the [`Final-Hybrid/`](Final-Hybrid/) folder and open the Jupyter notebooks
4. **Presentation** → View the [Final Review Presentation](Final%20Review%20PPT.pdf) for key insights

### Understanding the Code

- Open [`final-hybrid.ipynb`](Final-Hybrid/final-hybrid.ipynb) to see the model architecture and training process
- Open [`inference.ipynb`](Final-Hybrid/inference.ipynb) to see evaluation and example predictions
- Check [`app.py`](Final-Hybrid/app.py) for the interactive web interface implementation

### Exploring Development Process

- [`Phase_1/`](Phase_1/) contains initial research and baseline approaches
- [`Phase_2/`](Phase_2/) shows refinements and the final hybrid model selection

---

## Project Highlights

- **Addresses a specialized problem** in low-resource NLP: Sanskrit sandhi splitting
- **Combines multiple architectures** in a principled hybrid approach
- **Achieves strong performance** on Ayurvedic Sanskrit texts
- **Well-documented process** from exploration through final implementation
- **Interactive interface** available for live testing and demonstration

---

## Further Exploration

For questions about specific implementation details, architectural choices, or experimental results:
- Review the detailed methodology in the [Final Project Report](Final_Project_Report.pdf)
- Examine the code notebooks in [`Final-Hybrid/`](Final-Hybrid/)
- Check the phase documentation for development history and decisions

---

**Last Updated:** May 2026  
**Repository:** [Viserion-7/Final-Year-Project](https://github.com/Viserion-7/Final-Year-Project)
