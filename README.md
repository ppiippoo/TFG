# TFG

📦 Installation Instructions

To install the repository and the required programs, run the following command in a terminal (bash or cmd):

    $ git clone https://github.com/ppiippoo/TFG.git

🛠️ Main Libraries Used

    torch, torchvision — for deep learning, CNN models, and datasets

    lpips — for perceptual similarity

    matplotlib — for visualizations

    tqdm — for progress tracking

    opencv-python — for SIFT feature distance

    numpy, math — scientific and numerical support libraries

To install all required libraries:

    $ pip install torch torchvision lpips matplotlib tqdm opencv-python numpy

🚀 Running Experiments

Each experiment described in the thesis is implemented as a function in experiments.py.

Run any experiment from the root directory of the project (TFG) using:

python3 experiments.py <n> <model> [<threshold/removing_n>]

    <n> is an integer from 1 to 6, indicating the experiment number.

    <model> is the model used (e.g., alexnet or resnet-50).

    The third argument is only required for experiments 5 and 6:

        For experiment 5: provide a float between 0 and 1 (threshold).

        For experiment 6: provide an integer (number of nearest neighbors to remove).

🔍 Example Commands

- Clean training

      python3 experiments.py 1 alexnet
    
- Contamination with specific transformation
    
      python3 experiments.py 2 alexnet

- Contamination with near-duplicates

        python3 experiments.py 3 alexnet

- Contamination with exact duplicates

        python3 experiments.py 4 alexnet

- Decontamination using threshold

        python3 experiments.py 5 alexnet 0.5

- Decontamination by removing n near-duplicates

        python3 experiments.py 6 alexnet 10

    🧠 To use resnet-50 instead of alexnet, just replace "alexnet" with "resnet-50" in the command.

📌 Reproducibility Notes

To ensure consistent results, experiments were run using three different random seeds: 10, 11, and 12.
The average performance and standard deviation were reported.

    🛠️ Note: The seed value must be set manually at the beginning of the source code before executing the experiment.
