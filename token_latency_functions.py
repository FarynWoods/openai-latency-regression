import os
import time
from typing import Literal
from datetime import datetime
import ast

import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss, mean_squared_error


def get_token_count(text: str, model: str = "gpt-4o") -> int:
    """
    Return the number of tokens in the text.

    Parameters
    ----------
    text : str
        The input string.
    model : str, optional
        Any model name recognized by tiktoken. Defaults to "gpt-4o".
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback – same tokenizer family used by most ChatCompletion models
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def split_text_by_token_lengths(
    text: str, token_lengths: list, model="gpt-4o"
) -> list[str]:
    """Split a long string of text into a list of sub-strings with lengths
    given by token_lengths

    Parameters
    ----------
    text : str
        Long string of text to split
    token_lengths : list
        Desired token lengths for the resulting sub-strings
    model : str, optional
        Any model name recognized by tiktoken. Defaults to "gpt-4o".

    Returns
    -------
    list[str]
        List of sub-strings with lengths as defined by given token_lengths
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    prompts = []
    pos = 0

    for length in token_lengths:
        if pos + length > len(tokens):
            print("Ran out of words!")
            break
        chunk = tokens[pos : pos + length]  # noqa: E203
        prompts.append(enc.decode(chunk).strip())
        pos += length

    return prompts


def run_latency_experiments(
    prompts: list[str],
    *,
    client,
    model_name: str,
    iterations: int,
    csv_path: str,
    progress_every: int = 10,
) -> None:
    """Send each prompt to the given client and repeat for the given
    iterations, collecting timing and token data for each request.

    Parameters
    ----------
    prompts : list[str]
        List of prompts to send to the client
    client : openai.OpenAI
        OpenAI client
    model_name : str
        Specific model deployment to send requests to
    iterations : int
        How many times to repeat the same prompt request
    csv_path : str
        Path to save data
    progress_every : int, optional
        How often to print progress, by default 10
    """
    first_write = not os.path.exists(csv_path)
    fh = open(csv_path, "a")

    try:
        for p_idx, prompt in enumerate(prompts):
            print(f"Prompt {p_idx}: {iterations} iterations")

            for i in range(iterations):
                if i % progress_every == 0:
                    print(f"\tIteration {i}/{iterations}")

                row = _time_single_call(client, model_name, prompt)
                if row is None:
                    print("\t request returned no tokens; skipping")
                    continue

                pd.DataFrame([row]).to_csv(
                    fh,
                    header=first_write,
                    index=False,
                )
                first_write = False  # only write header once

    except KeyboardInterrupt:
        print("\nInterrupted — data up to this point is saved.")

    finally:
        fh.close()
        print(f"\nAll done. Data written to {csv_path}.")


def _time_single_call(client, model_name: str, prompt: str) -> dict:
    """Run one streamed completion and return timing/usage stats
    (or None on failure)."""
    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )

    token_times, tokens = [], []
    first_token_time = None

    input_toks = output_toks = reasoning_toks = cached_toks = None

    for chunk in response:
        now = time.time()

        if not chunk.choices:  # usage-only chunk
            if chunk.usage:
                input_toks = chunk.usage.prompt_tokens
                output_toks = chunk.usage.completion_tokens
                reasoning_toks = getattr(
                    chunk.usage.completion_tokens_details,
                    "reasoning_tokens",
                    None,
                )
                cached_toks = getattr(
                    chunk.usage.prompt_tokens_details, "cached_tokens", None
                )
            continue

        token = getattr(chunk.choices[0].delta, "content", "")
        if not token:
            continue

        if first_token_time is None:
            first_token_time = now
        token_times.append(now)
        tokens.append(token)

    if not token_times:  # request failed
        return None

    per_token_latency = [token_times[0] - start] + [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]

    return {
        "input_tokens": input_toks,
        "output_tokens": output_toks,
        "reasoning_tokens": reasoning_toks,
        "cached_tokens": cached_toks,
        "total_time": time.time() - start,
        "prefill_latency": token_times[0] - start,
        "token_latencies": per_token_latency,
    }


def fastest_input_token_times_fig(
    input_times: pd.DataFrame,
    model_name: str,
    network_latency: float = 0.0,
    quantile: float = 0.1,
) -> None:
    """Ignoring output tokens, plot the fastest 10% of recorded latencies for
    each input size.

    Parameters
    ----------
    input_times : pd.DataFrame
        DataFrame containing input token counts and recorded latencies
    model_name : str
        Model name to display in the figure title
    network_latency : float
        Trivial network latency to subtract from each point, defaults to 0.0
    quantile : float
        Which quantile and higher to plot
    """
    input_times["time_minus_network_latency"] = (
        input_times.prefill_latency - network_latency
    )
    fastest_10 = input_times.groupby("input_tokens", group_keys=False).apply(
        lambda group: group[
            group["time_minus_network_latency"]
            <= group["time_minus_network_latency"].quantile(quantile)
        ]
    )

    plt.scatter(
        fastest_10["input_tokens"],
        fastest_10["time_minus_network_latency"],
        alpha=0.5,
    )
    plt.title(
        f"Fastest {quantile*100}% Times per Input Token Count ({model_name})"
    )
    plt.xlabel("Input Tokens")
    plt.ylabel("Time (seconds)")
    plt.grid(True)


def fastest_output_token_times_fig(
    output_times: pd.DataFrame,
    output_bins: range,
    model_name: str,
    network_latency: float = 0.0,
    quantile: float = 0.1,
) -> None:
    """Ignoring input tokens, plot the fastest 10% of recorded latencies for
    each bin of output sizes.

    Parameters
    ----------
    output_times : pd.DataFrame
        DataFrame containing output token counts and recorded latencies
    output_bins : range
        Range of bin sizes, such as `range(0, 5000, 500)` for bins spanning 500
        tokens, up to an output size of 5000.
    model_name : str
        Model name to display in the figure title
    network_latency : float
        Trivial network latency to subtract from each point, defaults to 0.0
    quantile : float
        Which quantile and higher to plot
    """
    output_times["time_minus_network_latency"] = (
        output_times["total_time"] - network_latency
    )

    output_times["token_bin"] = pd.cut(
        output_times["output_tokens"], bins=output_bins
    )

    fastest_10 = output_times.groupby("token_bin", group_keys=False).apply(
        lambda g: g[
            g["time_minus_network_latency"]
            <= g["time_minus_network_latency"].quantile(quantile)
        ]
    )

    plt.figure()
    plt.scatter(
        fastest_10["output_tokens"],
        fastest_10["time_minus_network_latency"],
        alpha=0.5,
    )
    plt.title(
        f"Fastest {quantile*100}% Times per Output Token Count ({model_name})"
    )
    plt.xlabel("Output Tokens")
    plt.ylabel("Time (seconds)")
    plt.grid(True)


def plot_latency_contributions_by_term(
    input_tokens_map: dict, output_tokens: int, coeffs: pd.DataFrame
) -> None:
    """Show how each term of the regression contributes to the overall latency

    Parameters
    ----------
    input_tokens_map : dict
        _description_
    output_tokens : int
        _description_
    coeffs : pd.DataFrame
        _description_
    """
    output_k = output_tokens / 1000.0
    contrib_by_model = {}
    for model, input_list in input_tokens_map.items():
        row = coeffs[coeffs["model_name"] == model].iloc[0]
        const = row["quant_coef_const"]
        coef_in = row["quant_coef_in_k"]
        coef_out = row["quant_coef_out_k"]
        coef_int = row["quant_coef_in_out_k"]

        records = []
        for in_tok in input_list:
            in_k = in_tok / 1000.0
            c = const
            in_term = coef_in * in_k
            out_term = coef_out * output_k
            int_term = coef_int * in_k * output_k

            records.append(
                {
                    "input_tokens": in_tok,
                    "const": c,
                    "in_term": in_term,
                    "out_term": out_term,
                    "int_term": int_term,
                }
            )

        df_model = pd.DataFrame(records).set_index("input_tokens")
        contrib_by_model[model] = df_model

    num_models = len(contrib_by_model)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_models, figsize=(3 * num_models, 6), sharey=True
    )
    if num_models == 1:
        axes = [axes]

    for ax, (model, df_model) in zip(axes, contrib_by_model.items()):
        bottom = np.zeros(len(df_model))
        cmap = plt.cm.tab10
        term_colors = {
            "out_term": cmap(0),
            "int_term": cmap(1),
            "const": cmap(3),
            "in_term": cmap(2),
        }

        for col in ["const", "in_term", "out_term", "int_term"]:
            ax.bar(
                df_model.index.astype(str),
                df_model[col],
                bottom=bottom,
                label=col.replace("_", " ").title(),
                color=term_colors[col],
            )
            bottom += df_model[col].values

        ax.set_title(model)
        ax.set_xlabel("Input Tokens")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Latency (s)")
    axes[-1].legend(loc="upper right")

    fig.suptitle(
        f"Latency contributions for each regression term, output tokens = {output_tokens}"
    )

    plt.tight_layout()
    plt.show()

    return contrib_by_model


def preprocess_latency_data(
    df: pd.DataFrame,
    max_input_tokens: int,
    max_output_tokens: int,
) -> pd.DataFrame:
    """Filter outliers and scale the data to be in thousands of tokens, rather
    than single tokens, for numerical stability of regressions

    Parameters
    ----------
    df : pd.DataFrame
        data to preprocess and fit surface to
    max_input_tokens : int
        Give the actual max, or filter the data to just one part of the input
        token space
    max_output_tokens : int
        Give the actual max, or filter the data to just one part of the input
        token space

    Returns
    -------
    pd.DataFrame
        filtered and scaled dataframe
    """
    total_time_outliers = df["total_time"].quantile(0.995)
    network_adjusted_outliers = df["time_minus_network_latency"].quantile(0.99)
    filtered_df = df[
        (df["input_tokens"] < max_input_tokens)
        & (df["output_tokens"] < max_output_tokens)
        & (df["total_time"] <= total_time_outliers)
        & (df["time_minus_network_latency"] <= network_adjusted_outliers)
    ].copy()

    filtered_df["in_k"] = filtered_df.input_tokens / 1000.0
    filtered_df["out_k"] = filtered_df.output_tokens / 1000.0
    filtered_df["in_out_k"] = filtered_df["in_k"] * filtered_df["out_k"]
    return filtered_df


def compute_long_df(combined_data, network_latency):
    df = combined_data.copy()
    df["token_latencies"] = df["token_latencies"].apply(ast.literal_eval)
    df["obs_id"] = df.index
    long_df = (
        df.explode("token_latencies")
        .rename(columns={"token_latencies": "latency"})
        .reset_index(drop=True)
    )
    long_df["latency"] = long_df["latency"].astype(float)
    long_df["output_tokens"] = long_df.groupby("obs_id").cumcount() + 1
    long_df["total_time"] = long_df.groupby("obs_id")["latency"].cumsum()
    long_df["time_minus_network_latency"] = (
        long_df.total_time - network_latency
    )
    return long_df


def quantile_regression(
    df: pd.DataFrame,
    time_column: Literal[
        "time_minus_network_latency", "total_time"
    ] = "time_minus_network_latency",
    quantile: float = 0.10,
) -> tuple[RegressionResultsWrapper, float, float]:
    """Fit a quantile regression to predict the given quantile of the data, by
    default, the 10% quantile

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with in_k, out_k, and in_out_k columns
    time_column : str, optional
        Which latency column to fit to; either "total_time" or the default
        "time_minus_network_latency"
    quantile : float, optional
        quantile to fit regression to, by default 0.10, or the 10th quantile

    Returns
    -------
    RegressionResultsWrapper
        The regression results
    float
        The quantile/pinball loss of the model, based on the test set
    float
        Baseline loss on the test set
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = sm.add_constant(train_df[["in_k", "out_k", "in_out_k"]])
    y_train = train_df["time_minus_network_latency"]

    mod = sm.QuantReg(y_train, X_train)
    res = mod.fit(q=0.10)

    X_test = sm.add_constant(test_df[["in_k", "out_k", "in_out_k"]])
    y_test = test_df["time_minus_network_latency"]
    y_pred = res.predict(X_test)
    q_loss = mean_pinball_loss(y_test, y_pred, alpha=0.10)

    baseline_pred = np.full_like(y_test, y_train.quantile(0.10))
    baseline_loss = mean_pinball_loss(y_test, baseline_pred, alpha=0.10)

    print(res.summary())
    print("Quantile Loss on Test Set: ", q_loss)
    print(f"(Baseline Quantile Loss: {baseline_loss:.6f})")
    return res, q_loss, baseline_loss


def standard_linear_regression(
    df,
    time_column: Literal[
        "time_minus_network_latency", "total_time"
    ] = "time_minus_network_latency",
) -> tuple[RegressionResultsWrapper, float, float]:
    """Ordinary Least Squares regression of the data.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with in_k, out_k, and in_out_k columns
    time_column : str, optional
        Which latency column to fit to; either "total_time" or the default
        "time_minus_network_latency"

    Returns
    -------
    RegressionResultsWrapper
        The regression results
    float
        Mean squared error on the test set
    float
        Baseline mean squared error on the test set
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = sm.add_constant(train_df[["in_k", "out_k", "in_out_k"]])
    y_train = train_df[time_column]

    ols_mod = sm.OLS(y_train, X_train)
    ols_res = ols_mod.fit()

    X_test = sm.add_constant(test_df[["in_k", "out_k", "in_out_k"]])
    y_test = test_df[time_column]
    y_pred = ols_res.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    mean_baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = mean_squared_error(y_test, mean_baseline_pred)

    print(ols_res.summary())
    print(f"Mean Squared Error on Test Set: {mse:.6f}")
    print(f"(Baseline MSE: {baseline_mse:.6f})")
    return ols_res, mse, baseline_mse


def plot_3d(
    df: pd.DataFrame,
    model: RegressionResultsWrapper,
    title: str,
    observed_z_col: str | None = "time_minus_network_latency",
) -> go.Figure:
    """
    Plots a 3D surface and sample of observations using model coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'in_k', 'out_k', and optionally observed columns.
    model : RegressionResultsWrapper
        Fitted statsmodels regression model (OLS, QuantReg).
    title : str
        Plot title.
    observed_z_col : str or None
        If provided, used to plot actual sample points.

    Returns
    -------
    go.Figure
        The 3D surface plot figure.
    """
    in_k_vals = np.linspace(df.in_k.min(), df.in_k.max(), 60)
    out_k_vals = np.linspace(df.out_k.min(), df.out_k.max(), 60)
    in_grid, out_grid = np.meshgrid(in_k_vals, out_k_vals)

    flat_df = pd.DataFrame(
        {
            "in_k": in_grid.ravel(),
            "out_k": out_grid.ravel(),
        }
    )
    flat_df["in_out_k"] = flat_df["in_k"] * flat_df["out_k"]

    X_pred = sm.add_constant(flat_df)
    predicted = model.predict(X_pred)

    # Reshape to match grid
    lat_grid = predicted.values.reshape(in_grid.shape)

    # Convert axes back to tokens
    in_tokens = in_grid * 1000
    out_tokens = out_grid * 1000

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=in_tokens,
            y=out_tokens,
            z=lat_grid,
            colorscale="Viridis",
            opacity=0.85,
            showscale=True,
            colorbar=dict(title="Predicted latency (s)"),
            name="Fitted surface",
        )
    )

    # Optional observed points
    if observed_z_col is not None and observed_z_col in df.columns:
        sample = df.sample(n=min(5000, len(df)), random_state=0)
        fig.add_trace(
            go.Scatter3d(
                x=sample.input_tokens,
                y=sample.output_tokens,
                z=sample[observed_z_col],
                mode="markers",
                marker=dict(size=2, opacity=0.3),
                name="Observed",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Input tokens",
            yaxis_title="Output tokens",
            zaxis_title="Latency (s)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def save_latency_model_results(
    quant_result: RegressionResultsWrapper,
    quant_test_loss: float,
    quant_baseline_loss: float,
    ols_result: RegressionResultsWrapper,
    ols_test_loss: float,
    ols_baseline_loss: float,
    model_name: str,
    filename: str = "latency_coefficients.csv",
):
    """Save/append all regression results to file, by default
    'latency_coefficients.csv'"""

    timestamp = datetime.now().isoformat(timespec="seconds")

    def extract_result_data(result, prefix):
        data = {
            f"{prefix}_r_squared": (
                # Use pseudo rsquared if available, else use rsquared (OLS)
                getattr(result, "prsquared", None)
                if hasattr(result, "prsquared")
                else result.rsquared
            ),
            f"{prefix}_nobs": result.nobs,
        }
        for param in result.params.index:
            clean_param = param.replace(" ", "_")
            data[f"{prefix}_coef_{clean_param}"] = result.params[param]
            data[f"{prefix}_pval_{clean_param}"] = result.pvalues[param]
            data[f"{prefix}_se_{clean_param}"] = result.bse[param]
        return data

    quant_data = extract_result_data(quant_result, "quant")
    ols_data = extract_result_data(ols_result, "ols")

    row_data = {
        "model_name": model_name,
        "run_timestamp": timestamp,
        "quant_test_loss": quant_test_loss,
        "quant_baseline_loss": quant_baseline_loss,
        "ols_test_loss": ols_test_loss,
        "ols_baseline_loss": ols_baseline_loss,
    }
    row_data.update(quant_data)
    row_data.update(ols_data)

    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df
