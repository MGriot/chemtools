from scipy import stats


def t_students(alpha, d_f_):
    # Studnt, n= number of sample (n-1 degrees of freedom), p<0.05 (confidenza 1-alpha -> 1-0.05= 0.95, dato che è a due code si raddoppia il valore di alpha = (0.05/2)), 2-tail
    # Studnt, n=999, p<0.05%, Single tail
    t_one = stats.t.ppf(1 - alpha, d_f_)
    # print (stats.t.ppf(1-0.025, n))
    t_two = stats.t.ppf(1 - alpha / 2, d_f_)
    # print(f"tα/2: {t_two}")
    return t_one, t_two
