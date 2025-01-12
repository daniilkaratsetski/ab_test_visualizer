import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# We import Streamlit for creating interactive web apps,
# NumPy for mathematical operations and arrays,
# Matplotlib for data visualization,
# and the scipy.stats.norm for working with normal distributions.

# Page Configuration
# This sets the page title and layout for the Streamlit web application.
st.set_page_config(page_title="A/B Testing & Hypothesis Testing", layout="wide")

# Header
# Create the main title and initial description on the page.
st.title("A/B Testing & Hypothesis Testing")
st.write("Understand the basics of A/B testing and hypothesis testing through interactive visualizations and explanations.")

# Sample Size Calculator Section
# This section allows the user to input parameters for calculating sample size needed for a test.
st.header("Sample Size Calculator")
st.write(
    "### Hypothesis Testing Concepts\n"
    "- **Alpha (Type I error):** Probability of rejecting a true null hypothesis.\n"
    "- **Beta (Type II error):** Probability of failing to reject a false null hypothesis.\n"
    "- **Power (1 - Beta):** Probability of correctly rejecting a false null hypothesis.\n"
    "- **Effect Size:** Magnitude of the difference being tested.\n"
    "- **Sample Size:** Larger sample sizes reduce beta and increase power."
)

# These input fields collect user-defined parameters for calculating sample size:
#   mean1: Mean of sample 1
#   mean2: Mean of sample 2
#   alpha: Significance level (Type I error rate)
#   power: Probability (1 - beta) of correctly rejecting a false null hypothesis

st.sidebar.header("Main Parameters")
mean1 = st.sidebar.number_input("Enter mean of sample 1:", value=1.0, step=0.1)
mean2 = st.sidebar.number_input("Enter mean of sample 2:", value=2.0, step=0.1)
alpha = st.sidebar.number_input("Enter significance level (alpha):", value=0.05, min_value=0.01, max_value=0.99, step=0.01)
power = st.sidebar.number_input("Enter power (1 - beta):", value=0.8, min_value=0.1, max_value=0.99, step=0.01)

# Initialize shared variables in session_state for dynamic updates
# This ensures that the values persist even after user interactions.
if 'calculated_sample_size' not in st.session_state:
    st.session_state['calculated_sample_size'] = 100
if 'mean_diff' not in st.session_state:
    st.session_state['mean_diff'] = abs(mean1 - mean2)
if 'mean1' not in st.session_state:
    st.session_state['mean1'] = mean1
if 'mean2' not in st.session_state:
    st.session_state['mean2'] = mean2

# Function to calculate the required sample size based on:
#   - Z-values for the alpha (z_alpha) and beta (z_beta),
#   - effect_size = |mean1 - mean2|.
# It uses a simplified formula for sample size determination for two-sample testing.
def calculate_sample_size(mean1, mean2, alpha, power):
    effect_size = abs(mean1 - mean2)
    if effect_size == 0:
        return "Effect size cannot be zero. Please provide different mean values."
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    sample_size = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)
    return np.ceil(sample_size)

# When the user clicks the "Calculate Sample Size" button,
# we compute the required sample size and update the session state.
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

# Explanation of Accepting/Rejecting the Hypothesis
# This part discusses how to accept or reject a hypothesis based on p-values and confidence intervals.
st.header("Accepting/Rejecting the Hypothesis")

# Retrieve stored values from session state to keep consistency.
mean_diff = st.session_state['mean_diff']
mean1 = st.session_state['mean1']
mean2 = st.session_state['mean2']

# Users can input or adjust the sample size per group and confidence level.
sample_size = st.number_input("Sample size per group:", value=int(st.session_state['calculated_sample_size']), step=1)
confidence_level = st.slider("Confidence Level:", 0.9, 0.99, 0.95, step=0.01)

# Calculate standard error (SE). We're using a simplified approach (std dev = 1, so SE = 1/sqrt(n)).
standard_error = 1 / np.sqrt(sample_size)

# Calculate the z-value from the confidence level (two-sided).
z_value = norm.ppf(1 - (1 - confidence_level) / 2)

# Calculate confidence intervals for each mean:
ci1_lower = mean1 - z_value * standard_error
ci1_upper = mean1 + z_value * standard_error
ci2_lower = mean2 - z_value * standard_error
ci2_upper = mean2 + z_value * standard_error

# Calculate the test statistic and p-value for two-sided test:
z_stat = (mean1 - mean2) / standard_error
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

# Plot the distributions for both sample means with their confidence intervals.
x = np.linspace(-5, 5, 500)
y1 = norm.pdf(x, loc=mean1, scale=1)  # PDF for sample 1, std dev = 1
y2 = norm.pdf(x, loc=mean2, scale=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y1, label="Sample 1 Distribution", color="blue", alpha=0.8)
ax.plot(x, y2, label="Sample 2 Distribution", color="orange", alpha=0.8)

# Vertical lines show the confidence interval bounds for both samples.
ax.axvline(ci1_lower, color="blue", linestyle="--", label="Sample 1 CI Lower", alpha=0.8)
ax.axvline(ci1_upper, color="blue", linestyle="--", label="Sample 1 CI Upper", alpha=0.8)
ax.axvline(ci2_lower, color="orange", linestyle="--", label="Sample 2 CI Lower", alpha=0.8)
ax.axvline(ci2_upper, color="orange", linestyle="--", label="Sample 2 CI Upper", alpha=0.8)

ax.legend()
ax.set_title("Confidence Intervals for Two Distributions")
ax.set_xlabel("Value")
ax.set_ylabel("Probability Density")

# Display the plot in the Streamlit app.
st.pyplot(fig)

# We print out the results of our confidence intervals and p-value, guiding the user
# on how to interpret whether to reject the null hypothesis based on alpha vs p-value.
st.write(
    f"The confidence intervals for the two samples are:\n"
    f"- Sample 1: [{ci1_lower:.2f}, {ci1_upper:.2f}]\n"
    f"- Sample 2: [{ci2_lower:.2f}, {ci2_upper:.2f}]\n\n"
    f"The calculated p-value is: {p_value:.4f}.\n"
    f"If the p-value is less than the significance level (alpha), you reject the null hypothesis."
)


# New Section: Hypothesis Testing Visualizer: Power, Beta, and Sample Size
# A short introduction to key hypothesis testing concepts, including alpha, beta, and effect size.
st.header("Hypothesis Testing Visualizer: Power, Beta, and Sample Size")


# Sidebar Inputs for Graphs and Contours
#alpha = st.number_input("Alpha (𝛼) - significance level", value=0.05, min_value=0.01, max_value=0.1, step=0.01)
beta = -1*power + 1 #st.number_input("Beta (𝛽) - test power", value=0.8, min_value=0.5, max_value=0.99, step=0.01)
#mu_a = st.number_input("Mean for Sample A (𝜇𝐴)", value=1.0, step=0.1)
#mu_b = st.number_input("Mean for Sample B (𝜇𝐵)", value=1.1, step=0.1)


# Simulate A/B Tests
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

# Calculate minimum sample size
min_sample_size = int(calculate_sample_size( mean1, mean2, alpha, 1 - beta))
if isinstance(min_sample_size, str):
    st.error(min_sample_size)
else:
    # Input for sample size
    sample_size = st.number_input("Sample size (modifiable)", value=min_sample_size, step=1)

    # Run simulation
    results = simulate_tests(alpha, beta, mean1, mean2, sample_size)
    a_means, b_means, type_1_error, type_2_error = results

    # Plot histograms
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

    # Results Table
    st.subheader("Simulation Results")
    st.write(f"Minimum sample size: {min_sample_size}")
    if mean1 == mean2:
        st.write(f"Type I error: {type_1_error:.2%}")
    else:
        st.write(f"Type II error: {type_2_error:.2%}")


# Effect of Statistical Significance on Sample Size
# This section illustrates how alpha (significance level) affects required sample size.
st.header("Effect of Statistical Significance on Sample Size")

# Additional interactive inputs to showcase how changes in mean or power influence sample size
interactive_mean1 = st.number_input("Interactive Mean 1:", value=mean1, step=0.1)
interactive_mean2 = st.number_input("Interactive Mean 2:", value=mean2, step=0.1)
interactive_power = st.number_input("Interactive Power (1 - beta):", value=power, min_value=0.1, max_value=0.99, step=0.01)

# Generate a range of alpha values from 0.01 to 0.1
alpha_values = np.linspace(0.01, 0.1, 100)

# Calculate the required sample size for each alpha in that range,
# showing how stricter alpha levels (e.g., 0.01) demand larger samples.
required_samples = [calculate_sample_size(interactive_mean1, interactive_mean2, alpha, interactive_power) 
                    for alpha in alpha_values]

fig, ax = plt.subplots()
ax.plot(alpha_values, required_samples, label="Sample Size vs. Alpha")
ax.set_xlabel("Alpha Level (Significance)")
ax.set_ylabel("Required Sample Size")
ax.legend()
st.pyplot(fig)

st.write("As the significance level (alpha) decreases, the required sample size increases. This relationship highlights the tradeoff between statistical precision and sample size.")

# Effect of Sample Size with Beta
# This section focuses on how beta (Type II error) is related to sample size.
st.header("Effect of Sample Size on Beta")

interactive_alpha = st.number_input("Interactive Alpha (Significance Level):", value=alpha, min_value=0.01, max_value=0.1, step=0.01)
interactive_mean1_beta = st.number_input("Mean 1 for Beta Visualization:", value=mean1, step=0.1)
interactive_mean2_beta = st.number_input("Mean 2 for Beta Visualization:", value=mean2, step=0.1)

# Generate a range of beta values from 0.1 to 0.9 (thus power from 0.9 to 0.1),
# and compute sample size at each step to see how it scales.
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

