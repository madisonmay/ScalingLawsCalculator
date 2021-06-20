import streamlit as st
import string
import re
import math


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
N = st.sidebar.number_input(
    "Number of non-embedding params (millions):", 
    min_value=-1., 
    step=1.,
    format='%f',
    value=-1.,
)
D = st.sidebar.number_input(
    "Dataset size in tokens (millions):", 
    min_value=-1., 
    step=1.,
    format='%f',
    value=-1.,
)
C_min = st.sidebar.number_input(
    "Compute budget (PF-days)",
    min_value=-1.,
    step=1.,
    format='%e',
    value=-1.,
)
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
st.sidebar.markdown("## Power Laws ($\alpha$)")
a_N = st.sidebar.number_input("Params (N)", value=0.076)
a_D = st.sidebar.number_input("Dataset size (D)", value=0.095)
a_C_min = st.sidebar.number_input("Compute (C_min)", value=0.05)
st.sidebar.markdown("## Scale (tokenization-dependent)")
N_c = st.sidebar.number_input("Params (N)", value=8.8e13, format='%e')
D_c = st.sidebar.number_input("Dataset size (D)", value=5.4e13, format='%e')
C_min_c = st.sidebar.number_input("Compute (C_min)", value=3.1e8, format='%e')

st.markdown("## Compute Efficient Frontier")
st.markdown("Specify your compute budget to determine optimal settings.")
col0, col1, col2 = st.beta_columns(3)
col1.markdown("**Formula**")
col2.markdown("**Result**")

# Compute optimal param count given fixed compute budget
N_opt = N_e * C_min ** p_N
N_opt_formatted = fmt.format("{:s}", N_opt) if C_min > 0 else "N_{opt}" 
C_formatted = fmt.format("({:s})", C_min) if C_min > 0 else "C_{min}"

col0.markdown("**Computed Value**")
col0.markdown("Optimal parameter count (non-embedding)")
col1.markdown("$$N_{opt} = N_e \cdot C_{min}^{p_N}$$")
col2.markdown("$$" + N_opt_formatted + fmt.format(" = {:s} \cdot ", N_e) + C_formatted + "^{" + f"{p_N:.02f}" + "}$$")

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
col1.markdown("$$D_{opt} = D_e \cdot C_{min}^{p_D}$$")
col2.markdown("$$" + D_opt_formatted + fmt.format(" = {:s} \cdot ", D_e) + C_formatted + "^{" + f"{p_D:.02f}" + "}$$")
