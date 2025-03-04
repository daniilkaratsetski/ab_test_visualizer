import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="A/B Testing & Hypothesis Testing", layout="centered")


st.title("A/B Testing & Hypothesis Testing")
st.write("Understand the basics of A/B testing and hypothesis testing through interactive visualizations and explanations.")


st.header("Sample Size Calculator")
st.write(
    "### Hypothesis Testing Concepts\n"
    "- **Alpha (Type I error):** Probability of rejecting a true null hypothesis.\n"
    "- **Beta (Type II error):** Probability of failing to reject a false null hypothesis.\n"
    "- **Power (1 - Beta):** Probability of correctly rejecting a false null hypothesis.\n"
    "- **Effect Size:** Magnitude of the difference being tested.\n"
    "- **Sample Size:** Larger sample sizes reduce beta and increase power."
)

st.sidebar.header("Main Parameters")
mean1 = st.sidebar.number_input("Enter mean of sample 1:", value=1.0, step=0.1)
mean2 = st.sidebar.number_input("Enter mean of sample 2:", value=1.1, step=0.1)
alpha = st.sidebar.number_input("Enter significance level (alpha):", value=0.05, min_value=0.01, max_value=0.99, step=0.01)
power = st.sidebar.number_input("Enter power (1 - beta):", value=0.8, min_value=0.1, max_value=0.99, step=0.01)


if 'calculated_sample_size' not in st.session_state:
    st.session_state['calculated_sample_size'] = 100
if 'mean_diff' not in st.session_state:
    st.session_state['mean_diff'] = abs(mean1 - mean2)
if 'mean1' not in st.session_state:
    st.session_state['mean1'] = mean1
if 'mean2' not in st.session_state:
    st.session_state['mean2'] = mean2

def calculate_sample_size(mean1, mean2, alpha, power):
    effect_size = abs(mean1 - mean2)
    if effect_size == 0:
        return "Effect size cannot be zero. Please provide different mean values."
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    sample_size = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)
    return np.ceil(sample_size)

if st.button("Calculate Sample Size"):
    sample_size = calculate_sample_size(mean1, mean2, alpha, power)
    if isinstance(sample_size, str):  # If there's an error message
        st.error(sample_size)
    else:
        st.session_state['calculated_sample_size'] = sample_size
        st.session_state['mean_diff'] = abs(mean1 - mean2)
        st.session_state['mean1'] = mean1
        st.session_state['mean2'] = mean2
        st.success(f"Required Sample Size: {sample_size}")
        st.write("This calculation helps determine the number of observations needed to detect a statistically significant difference between the two sample means.")


st.header("Accepting/Rejecting the Hypothesis")

mean_diff = st.session_state['mean_diff']
mean1 = st.session_state['mean1']
mean2 = st.session_state['mean2']

sample_size = st.number_input("Sample size per group:", value=int(st.session_state['calculated_sample_size']), step=1)
confidence_level = st.slider("Confidence Level:", 0.9, 0.99, 0.95, step=0.01)

standard_error = 1 / np.sqrt(sample_size)

z_value = norm.ppf(1 - (1 - confidence_level) / 2)

ci1_lower = mean1 - z_value * standard_error
ci1_upper = mean1 + z_value * standard_error
ci2_lower = mean2 - z_value * standard_error
ci2_upper = mean2 + z_value * standard_error

z_stat = (mean1 - mean2) / standard_error
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

x = np.linspace(-5, 5, 500)
y1 = norm.pdf(x, loc=mean1, scale=1)  # PDF for sample 1, std dev = 1
y2 = norm.pdf(x, loc=mean2, scale=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y1, label="Sample 1 Distribution", color="blue", alpha=0.8)
ax.plot(x, y2, label="Sample 2 Distribution", color="orange", alpha=0.8)

ax.axvline(ci1_lower, color="blue", linestyle="--", label="Sample 1 CI Lower", alpha=0.8)
ax.axvline(ci1_upper, color="blue", linestyle="--", label="Sample 1 CI Upper", alpha=0.8)
ax.axvline(ci2_lower, color="orange", linestyle="--", label="Sample 2 CI Lower", alpha=0.8)
ax.axvline(ci2_upper, color="orange", linestyle="--", label="Sample 2 CI Upper", alpha=0.8)

ax.legend()
ax.set_title("Confidence Intervals for Two Distributions")
ax.set_xlabel("Value")
ax.set_ylabel("Probability Density")

st.pyplot(fig)


st.write(
    f"The confidence intervals for the two samples are:\n"
    f"- Sample 1: [{ci1_lower:.2f}, {ci1_upper:.2f}]\n"
    f"- Sample 2: [{ci2_lower:.2f}, {ci2_upper:.2f}]\n\n"
    f"The calculated p-value is: {p_value:.4f}.\n"
    f"If the p-value is less than the significance level (alpha), you reject the null hypothesis."
)


st.header("Hypothesis Testing Visualizer: Power, Beta, and Sample Size")


#alpha = st.number_input("Alpha (𝛼) - significance level", value=0.05, min_value=0.01, max_value=0.1, step=0.01)
beta = -1*power + 1 #st.number_input("Beta (𝛽) - test power", value=0.8, min_value=0.5, max_value=0.99, step=0.01)
#mu_a = st.number_input("Mean for Sample A (𝜇𝐴)", value=1.0, step=0.1)
#mu_b = st.number_input("Mean for Sample B (𝜇𝐵)", value=1.1, step=0.1)


def simulate_tests(alpha, beta, mean1, mean2, sample_size):
    a_samples = np.random.normal(mean1, 1, (1000, sample_size))
    b_samples = np.random.normal(mean2, 1, (1000, sample_size))

    a_means = a_samples.mean(axis=1)
    b_means = b_samples.mean(axis=1)

    t_statistics = (b_means - a_means) / np.sqrt((1 / sample_size) * 2)
    p_values = norm.sf(t_statistics)

    type_1_error = np.mean(p_values < alpha) if mean1 == mean2 else None
    type_2_error = np.mean(p_values >= alpha) if mean1 != mean2 else None

    return a_means, b_means, type_1_error, type_2_error

min_sample_size = int(calculate_sample_size( mean1, mean2, alpha, 1 - beta))
if isinstance(min_sample_size, str):
    st.error(min_sample_size)
else:
    sample_size = st.number_input("Sample size (modifiable)", value=min_sample_size, step=1)

    results = simulate_tests(alpha, beta, mean1, mean2, sample_size)
    a_means, b_means, type_1_error, type_2_error = results

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(a_means, bins=30, alpha=0.7, label="Sample A")
    ax.hist(b_means, bins=30, alpha=0.7, label="Sample B")
    critical_value = mean1 + norm.ppf(1 - alpha) * (1 / np.sqrt(sample_size))
    ax.axvline(x=critical_value, color='red', linestyle='--', label="Alpha (significance level)")
    ax.legend()
    ax.set_title("Histograms of Sample Means")
    ax.set_xlabel("Mean Values")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Simulation Results")
    st.write(f"Minimum sample size: {min_sample_size}")
    if mean1 == mean2:
        st.write(f"False Positive error: {type_1_error:.2%}")
    else:
        st.write(f"False Negative error: {type_2_error:.2%}")



st.header("Effect of Statistical Significance on Sample Size")

interactive_mean1 = st.number_input("Interactive Mean 1:", value=mean1, step=0.1)
interactive_mean2 = st.number_input("Interactive Mean 2:", value=mean2, step=0.1)
interactive_power = st.number_input("Interactive Power (1 - beta):", value=power, min_value=0.1, max_value=0.99, step=0.01)

alpha_values = np.linspace(0.01, 0.1, 100)


required_samples = [calculate_sample_size(interactive_mean1, interactive_mean2, alpha, interactive_power) 
                    for alpha in alpha_values]

fig, ax = plt.subplots()
ax.plot(alpha_values, required_samples, label="Sample Size vs. Alpha")
ax.set_xlabel("Alpha Level (Significance)")
ax.set_ylabel("Required Sample Size")
ax.legend()
st.pyplot(fig)

st.write("As the significance level (alpha) decreases, the required sample size increases. This relationship highlights the tradeoff between statistical precision and sample size.")

st.header("Effect of Sample Size on Beta")

interactive_alpha = st.number_input("Interactive Alpha (Significance Level):", value=alpha, min_value=0.01, max_value=0.1, step=0.01)
interactive_mean1_beta = st.number_input("Mean 1 for Beta Visualization:", value=mean1, step=0.1)
interactive_mean2_beta = st.number_input("Mean 2 for Beta Visualization:", value=mean2, step=0.1)

beta_values = np.linspace(0.1, 0.9, 100)
required_samples_beta = [
    calculate_sample_size(interactive_mean1_beta, interactive_mean2_beta, interactive_alpha, 1 - beta)
    for beta in beta_values
]

fig, ax = plt.subplots()
ax.plot(beta_values, required_samples_beta, label="Sample Size vs. Beta")
ax.set_xlabel("Beta (1 - Power)")
ax.set_ylabel("Required Sample Size")
ax.set_title("Effect of Beta on Required Sample Size")
ax.legend()
st.pyplot(fig)

st.write(
    "As beta decreases (or power increases), the required sample size grows. "
    "This demonstrates the tradeoff between desired power and the number of observations needed."
)

