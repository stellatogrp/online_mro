"""MOSEK solver options shared by every SPARSE-SVM experiment solve.

Provenance: verbatim copy of the solver constants from legacy
``svm/utils.py`` (which ``svm1/utils.py`` re-exported unchanged).
"""

# MOSEK solve options shared by every experiment solve: a hard 1500 s wall-clock
# cap on both the continuous and mixed-integer optimizers (returns the best
# incumbent found so far if the cap is hit).
MOSEK_TIME_LIMIT = 1500.0
MOSEK_PARAMS = {
    'MSK_DPAR_OPTIMIZER_MAX_TIME': MOSEK_TIME_LIMIT,
    'MSK_DPAR_MIO_MAX_TIME': MOSEK_TIME_LIMIT,
    # One thread per solve: the experiments fan out R*M joblib workers across the
    # allocated cores, so letting each MOSEK solve grab all cores oversubscribes
    # the CPU and slows every worker. See PROFILING.md.
    'MSK_IPAR_NUM_THREADS': 1,
}
