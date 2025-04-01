# Fast Data-Driven Model Predictive Control via Mean Robust Optimization
This repository is by
[Irina Wang](https://sites.google.com/view/irina-wang),
Marta Fochesato,
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"Data Compression for Fast Online Stochastic Optimization."

If you find this repository helpful in your publications,
please consider citing our paper.

## Introduction
We propose an online data compression approach for efficiently solving DRO problems with streaming data while maintaining out-of-sample performance guarantees. Our method dynamically constructs ambiguity sets using online clustering, allowing the clustered configuration to evolve over time for an accurate representation of the underlying distribution. We establish theoretical conditions for clustering algorithms to ensure robustness, and show that the performance gap between our online solution and the nominal DRO solution is controlled by the Wasserstein distance between the true and compressed distributions, which is approximated using empirical measures. 
We provide a regret analysis, proving that the upper bound on this performance gap converges sublinearly to a fixed 
clustering-dependent distance, even when nominal DRO has access,  in hindsight, to the subsequent realization of the uncertainty.
Numerical experiments in mixed-integer portfolio optimization demonstrate significant computational savings, with minimal loss in solution quality.

## Dependencies
Install dependencies with
```
pip install -r requirements.txt
```

## Instructions
### Running experiments
A valid mosek license is required in the mosek folder.
Experiments can from the root folder using the commands below. The value R controls the number of repetitions.
If you wish to run experiments in separate batches, you can set r_start to the cumulative total already ran. 
E.g. r_start 0 with R = 10 is equivalent to running R = 5 twice, with r_start = 0 then r_start 5. 

```
python portfolio/portMIP.py --R 10 --K 5  --T 3001 --fixed_time 2500 --interval 100 --Q 500 --N_init 50 --r_start 0
python portfolio/portMIP.py  --R 10 --K 15 --T 3001 --fixed_time 2500 --interval 100 --Q 500 --N_init 50 --r_start 0
python portfolio/portMIP_DRO.py --R 10 --T 3001 --interval 100 --interval_SAA 100 --N_init 50 --r_start 0
python portfolio/portMIP.py --R 10 --K 15 --T 10001 --fixed_time 7000  --interval 100 --Q 500 --N_init 50 --r_start 0 
```

After running the above, plots can be generated using the following command.

```
python portfolio/plots.py --foldername portfolio_exp/ --R 10
```

