# Transformers-FI
Analysis of the efficiency of activation value clipping as a fault-tolerance mechanism for Transformers models. 

## Structure
This repository contains 2 parts:
- `activation_analysis/` folder that contains the code used to perform the analysis on Transformers' activations and generate Figures 2 and 3.
- `transformers_fi/` folder that contain the code used to perform de the fault injection campaings and to generate Figure 4.

## Reproducability
To allow reproducability of the different figures, all the needed data are available on this repository.

First, you need to clone the repository.

```bash
git clone https://github.com/lucasrqt/Transformers-FI.git
cd Transformers-FI/
```

### Figures 2 and 3
The data needed for Figures 2 and 3 are stored in `activation_analysis/data/`.

The different datas are stored in NumPy files for each model.

#### Figures generation
```bash
cd activation_analysis # assuming you are already in the root folder of this repository

# ensure you can execute the script
chmod u+x plot_activation_data.py 

# run the script
./plot_activation_data.py 
```

After the execution of the script, the figures are stored in `activation_analysis/data/plots`.

### Figure 4

The results of the injection campaings are stored in the folder `transformers_fi/results/`, and split in folders (one folder per assessed model).

#### Figure generation

```bash
cd transformers_fi # assuming you are already in the root folder of this repository

# ensure you can execute the script
chmod u+x parse_results.py 

# run the script
./parse_results.py 
```

By default, the results and plot are stored in `parsed_results`. 

For more information, run:
```bash
./parse_results.py --help
```

## Disclaimer

The code-base comes from an another project. Therefore some elements are still present but not used.

## Contact
Don't hesitate to contact me for any question at `lucas.roquet@inria.fr`. 