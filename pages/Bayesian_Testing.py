import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

st.set_page_config(layout="centered", page_title="Bayesian Updating with Plotly")

st.title("Bayesian Updating with Plotly")
st.markdown("""
This example demonstrates how the posterior distribution updates with each new observation.
Select prior distribution parameters, the true probability of heads, and the total number of observations.
All relevant information is displayed in the legend on the right.
""")

st.sidebar.header("Set Parameters")
alpha_prior = st.sidebar.slider("Alpha (heads) - prior", min_value=1, max_value=100, value=50)
beta_prior = st.sidebar.slider("Beta (tails) - prior", min_value=1, max_value=100, value=50)
real_p = st.sidebar.slider("True probability of heads", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
n_observations = st.sidebar.number_input("Total number of observations", min_value=1, max_value=1000, value=100)

np.random.seed(42)
observations = np.random.choice([0, 1], size=int(n_observations), p=[1 - real_p, real_p])

step = st.slider("Number of considered observations", min_value=0, max_value=int(n_observations), value=2, step=1)

current_data = observations[:step]
heads_count = np.sum(current_data)
tails_count = step - heads_count

alpha_post = alpha_prior + heads_count
beta_post = beta_prior + tails_count

x = np.linspace(0, 1, 500)
prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(x, alpha_post, beta_post)

prior_mean = alpha_prior / (alpha_prior + beta_prior)
posterior_mean = alpha_post / (alpha_post + beta_post)
overlap = np.trapz(np.minimum(prior_pdf, posterior_pdf), x)
p_better = 1 - beta.cdf(prior_mean, alpha_post, beta_post)
p_worse = beta.cdf(prior_mean, alpha_post, beta_post)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=prior_pdf, mode="lines", name="Prior Distribution", line=dict(dash="dash", color="orange")))
fig.add_trace(go.Scatter(x=x, y=posterior_pdf, mode="lines", name="Posterior Distribution", line=dict(color="blue")))

fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"Total observations: {step}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"Heads (1): {heads_count}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"Tails (0): {tails_count}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"Prior mean: {prior_mean:.3f}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"p(posterior > {prior_mean:.3f}): {p_better:.3f}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"p(posterior < {prior_mean:.3f}): {p_worse:.3f}"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, opacity=0), showlegend=True, name=f"Overlap: {overlap:.3f}"))

fig.update_layout(title=f"Beta Distribution Update: {step} Observations", xaxis_title="Probability of Heads", yaxis_title="Density", margin=dict(l=50, r=300, t=50, b=50), legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top'))

fig.add_annotation(x=posterior_mean, y=beta.pdf(posterior_mean, alpha_post, beta_post), text=f"Posterior Mean: {posterior_mean:.3f}", showarrow=True, arrowhead=2)

st.plotly_chart(fig)
