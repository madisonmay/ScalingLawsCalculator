import streamlit as st
import string
import re
import matplotlib.pyplot as plt
import numpy as np


PF_day = 8.64e19

class ScientificNotationFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if format_spec == 's':
            default_fmt = super().format_field(value, '.02e')
            default_fmt = default_fmt.replace('e+', '\cdot 10^{').replace('e-', '\cdot 10^{-')+ "}"
            return re.sub(r'0(?=\d+)', '', default_fmt)
        else:
            return super().format_field(value, format_spec)

fmt = ScientificNotationFormatter()

st.set_page_config(layout="wide")
st.sidebar.markdown("## Variables")
st.sidebar.markdown("Value of -1 indicates unspecified")
C_min = st.sidebar.number_input(
    "Compute budget (PF-days)",
    min_value=-1.,
    step=1.,
    format='%f',
    value=-1.,
)
N = st.sidebar.number_input(
    "Number of non-embedding params (millions):", 
    min_value=-1., 
    step=1.,
    format='%f',
    value=-1.,
)
N = N * 1e6 if N > 0 else N
D = st.sidebar.number_input(
    "Dataset size in tokens (millions):", 
    min_value=-1., 
    step=1.,
    format='%f',
    value=-1.,
)
D = D * 1e6 if D > 0 else D

st.sidebar.markdown("## Power Laws for Compute Efficient Training")
p_N = st.sidebar.number_input("Params (N)", value=0.73)
p_B = st.sidebar.number_input("Batch size (B)", value=0.24)
p_S = st.sidebar.number_input("Steps (S)", value=0.03)
p_D = st.sidebar.number_input("Dataset size (D)", value=0.27)
st.sidebar.markdown("## Scale for Compute Efficient Training")
N_e = st.sidebar.number_input("Params (N)", value=1.3e9, format='%e')
B_e = st.sidebar.number_input("Batch size (B)", value=2.0e6, format='%e')
S_e = st.sidebar.number_input("Steps (S)", value=5.4e3, format='%e')
D_e = st.sidebar.number_input("Dataset size (D)", value=2.0e10, format='%e')
st.sidebar.markdown("## Power Laws")
a_N = st.sidebar.number_input("Params (N)", value=0.076)
a_D = st.sidebar.number_input("Dataset size (D)", value=0.095)
a_C_min = st.sidebar.number_input("Compute (C_min)", value=0.05)
a_S = st.sidebar.number_input("Steps (S)", value=0.76)
st.sidebar.markdown("## Scale (tokenization-dependent)")
N_c = st.sidebar.number_input("Params (N)", value=8.8e13, format='%e')
D_c = st.sidebar.number_input("Dataset size (D)", value=5.4e13, format='%e')
C_min_c = st.sidebar.number_input("Compute (C_min)", value=3.1e8, format='%e')
S_c = st.sidebar.number_input("Steps (S)", value=2.1e3, format='%e')

st.markdown("## Compute Efficient Frontier")
st.markdown("**Specify your compute budget to determine optimal settings.**")
col0, col1, col2 = st.beta_columns(3)
col1.markdown("**Formula**")
col2.markdown("**Result**")

D_formatted = fmt.format("{:s}", D) if D > 0 else "D" 
N_formatted = fmt.format("{:s}", N) if N > 0 else "N"

# Compute optimal param count given fixed compute budget
N_eff = N_e * C_min ** p_N
N_eff_formatted = fmt.format("{:s}", N_eff) if C_min > 0 else "N_{eff}" 
C_formatted = fmt.format("({:s})", C_min) if C_min > 0 else "C_{min}"

col0.markdown("**Computed Value**")
col0.markdown("Optimal param count (non-embedding)")
col1.markdown("$$N_{eff} = N_e \cdot C_{min}^{p_N}$$")
col2.markdown("$$" + N_eff_formatted + fmt.format(" = {:s} \cdot ", N_e) + C_formatted + "^{" + f"{p_N:.02f}" + "}$$")

# Critical batch size
B_crit = B_e * C_min ** p_B
B_crit_formatted = fmt.format("{:s}", B_crit) if C_min > 0 else "B_{crit}" 

col0.markdown("Critical batch size (tokens).  Ensure $B \ll B_{crit}$")
col1.markdown("$$B_{crit} = B_e \cdot C_{min}^{p_B}$$")
col2.markdown("$$" + B_crit_formatted + fmt.format(" = {:s} \cdot ", B_e) + C_formatted + "^{" + f"{p_B:.02f}" + "}$$")


# Critical batch size
S_min = S_e * C_min ** p_S
S_min_formatted = fmt.format("{:s}", S_min) if C_min > 0 else "S_{min}" 

col0.markdown("Lower bound on number of steps")
col1.markdown("$$S_{min} = S_e \cdot C_{min}^{p_S}$$")
col2.markdown("$$" + S_min_formatted + fmt.format(" = {:s} \cdot ", S_e) + C_formatted + "^{" + f"{p_S:.02f}" + "}$$")

# Critical batch size
D_opt = D_e * C_min ** p_D
D_opt_formatted = fmt.format("{:s}", D_opt) if C_min > 0 else "D_{opt}" 

col0.markdown("Optimal dataset size (tokens)")
col1.markdown("$$D_{eff} = D_e \cdot C_{min}^{p_D}$$")
col2.markdown("$$" + D_opt_formatted + fmt.format(" = {:s} \cdot ", D_e) + C_formatted + "^{" + f"{p_D:.02f}" + "}$$")


# Regime
if N > 0 and D > 0:
    if (D_c * D ** a_D) < (N_c * N ** a_N):
        regime = "*Primarily Data-limited*"
    else:
        regime = "*Primarily Capacity-limited*"

    st.markdown(f"## Regime: {regime}")
else:
    st.markdown(f"## Regime")
    st.markdown(f"**Specify param count and dataset size to determine if you are data-limited or capacity-limited.**")

if N > 0 and C_min > 0:
    st.markdown(f"Param count ($$" +  N_formatted + f"$$) is {'high' if N > N_eff else 'low'} relative to "+ "$$N_{eff}$$ of $$" + N_eff_formatted + "$$")
if D > 0 and C_min > 0:
    st.markdown(f"Token count ($$" + D_formatted + f"$$) is {'high' if D > D_opt else 'low'} relative to " + "$$D_{eff}$$ of $$" + D_opt_formatted + "$$")


col0, col1 = st.beta_columns(2)
if C_min > 0 and N > 0:
    col0.markdown("## Extra-compute due to sub-optimal N")
    col0.markdown("If your $$N$$ doesn't appear on this plot your model is too small for your compute budget.")
    col0.markdown("Red: $$N$$, Green: $$N_{eff}$$")
    fig = plt.figure()
    n = N_eff * np.logspace(start=-2, stop=2)
    compute_use = (n / N_eff) *  (1 + (a_S/a_N)*(1 - (N_eff / n) ** a_N)) ** (-1/a_S)
    # Equation can require taking fractional exponent of negative number.
    # Currently assuming eqn is ill defined for those values.
    valid_n = np.argwhere(~np.isnan(compute_use))
    n, compute_use = n[valid_n], compute_use[valid_n] 
    plt.plot(n, compute_use)
    plt.plot(N_eff, 1, marker='o', color='g')
    if np.min(n) < N < np.max(n):
        C_x = (N / N_eff) *  (1 + (a_S/a_N)*(1 - (N_eff / N) ** a_N)) ** (-1/a_S)
        plt.plot([N], [C_x], marker='o', color='r')
    plt.xlabel("Non-embedding paramater count")
    plt.ylabel("Ratio of compute use to optimal")
    plt.xscale("log")
    col0.pyplot(fig)
