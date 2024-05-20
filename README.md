# Code accompanying the paper: A model of how hierarchical representations constructed in hippocampus are used to navigate through space

# Organization
#### state_abstraction_rl/state_abstraction/
Contains code for creating the hierarchical abstractions given a known environment. The class AbstractedMDP is the core class, and implements algorithm #2 from the paper. This class relies on the [Hippocluster library](https://github.com/echalmers/hippocluster) 

#### state_abstraction_rl/agents/
Contains learning agent classes. MBRL is the basic model-based learning agent described in algorithm #1 from the paper. *H*MBRL subclasses MBRL, and implements a learning agent that creates its own hierarchical abstractions of an environment. It aggregates an AbstractedMDP object to create hierarchical abstractions, and ValueIterationPlanner objects to effect the hierarchical planning illustrated in algorithm #3 from the paper.

#### scripts/
Run the hierarchical_demo.py script to get a visual demo of the hierarchical agent in action (see steps below to get started). The experiments/ folder contains scripts used to create figures for the paper. These scripts generally come in pairs: e.g. cognitive_load.py runs the cognitive load experiment and saves the results, while cognitive_load_results.py generates charts from the saved data.

# How to run

To run scripts from this repository, you can either use the PyCharm IDE and run them directly from within it or you can setup a custom virtual environment and run them via the command line. Below are some instructions on how to setup the custom environment and run a demo script[^1].

### Custom env Instructions

1. Create a virtual environment

```py
python3 -m venv .env
```

2. Enable the previously created environment

```py
source .env/bin/activate
```

3. Install dependencies and requirements
```py
pip3 install -e .
```

4. Ensure it is working by running the follow command
```py
python3 scripts/hierarchical_demo.py
```

[^1]: These commands were tested on Mac, and may vary on a Windows machine.
