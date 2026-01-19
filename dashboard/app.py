"""
Legal Attitudes LLM Dashboard

Visualize model responses to legal attitude scales.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
REFUSAL_CODE = 7
DATA_DIR = Path(__file__).parent.parent / "data"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Model ID to display label mapping
MODEL_LABELS = {
    # OpenAI
    "gpt-4o-2024-11-20": "GPT-4o",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-5-2025-08-07": "GPT-5",
    "gpt-5.1-2025-11-13": "GPT-5.1",
    "gpt-5.2-2025-12-11": "GPT-5.2",
    # Anthropic
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-haiku-4-5-20251001": "Claude 4.5 Haiku",
    "claude-sonnet-4-20250514": "Claude 4 Sonnet",
    "claude-sonnet-4-5-20250929": "Claude 4.5 Sonnet",
    "claude-opus-4-5-20251101": "Claude 4.5 Opus",
    # Google
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    # DeepSeek
    "deepseek_deepseek-chat-v3.1": "DeepSeek Chat v3.1",
    "deepseek_deepseek-r1": "DeepSeek R1",
    "deepseek_deepseek-v3.2": "DeepSeek v3.2",
    # Meta Llama
    "meta-llama_llama-3-70b-instruct": "Llama 3 70B",
    "meta-llama_llama-3.1-70b-instruct": "Llama 3.1 70B",
    "meta-llama_llama-3.1-405b-instruct": "Llama 3.1 405B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    # Qwen
    "qwen_qwen3-235b-a22b-2507": "Qwen3 235B",
    "qwen_qwen3-max": "Qwen3 Max",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}

# Provider color palette
PROVIDER_COLORS = {
    "openai": "#10a37f",      # OpenAI green
    "anthropic": "#d4a574",   # Anthropic tan/gold
    "google": "#4285f4",      # Google blue
    "deepseek": "#7c3aed",    # DeepSeek purple
    "meta": "#0668e1",        # Meta blue
    "qwen": "#ff6b35",        # Qwen orange
}


def get_provider_from_model(model_id: str) -> str:
    """Get provider name from model ID."""
    if model_id.startswith("gpt-"):
        return "openai"
    elif model_id.startswith("claude-"):
        return "anthropic"
    elif model_id.startswith("gemini-"):
        return "google"
    elif model_id.startswith("deepseek"):
        return "deepseek"
    elif model_id.startswith("meta-llama"):
        return "meta"
    elif model_id.startswith("qwen"):
        return "qwen"
    return "unknown"


def get_model_color(model_id: str) -> str:
    """Get color for a model based on its provider."""
    provider = get_provider_from_model(model_id)
    return PROVIDER_COLORS.get(provider, "#888888")


# Model capability ordering (weaker to stronger within each provider)
# Lower number = weaker, higher number = stronger
MODEL_CAPABILITY_ORDER = {
    # OpenAI (0-99)
    "gpt-4o-2024-11-20": 0,
    "gpt-4.1-2025-04-14": 1,
    "gpt-5-2025-08-07": 2,
    "gpt-5.1-2025-11-13": 3,
    "gpt-5.2-2025-12-11": 4,
    # Anthropic (100-199)
    "claude-3-7-sonnet-20250219": 100,
    "claude-haiku-4-5-20251001": 101,
    "claude-sonnet-4-20250514": 102,
    "claude-sonnet-4-5-20250929": 103,
    "claude-opus-4-5-20251101": 104,
    # Google (200-299)
    "gemini-2.0-flash": 200,
    "gemini-2.5-flash": 201,
    "gemini-2.5-pro": 202,
    "gemini-3-flash-preview": 203,
    "gemini-3-pro-preview": 204,
    # DeepSeek (300-399)
    "deepseek_deepseek-chat-v3.1": 300,
    "deepseek_deepseek-r1": 301,
    "deepseek_deepseek-v3.2": 302,
    # Meta Llama (400-499)
    "meta-llama_llama-3-70b-instruct": 400,
    "meta-llama_llama-3.1-70b-instruct": 401,
    "meta-llama_llama-3.3-70b-instruct": 402,
    "meta-llama_llama-3.1-405b-instruct": 403,
    # Qwen (500-599)
    "qwen_qwen3-next-80b-a3b-thinking": 500,
    "qwen_qwen3-235b-a22b-2507": 501,
    "qwen_qwen3-max": 502,
}


def get_model_sort_key(model_id: str) -> int:
    """Get sort key for model based on capability (weaker to stronger)."""
    return MODEL_CAPABILITY_ORDER.get(model_id, 9999)


def get_model_label_sort_key(model_label: str) -> int:
    """Get sort key for model label based on capability."""
    # Reverse lookup: find model_id from label
    for model_id, label in MODEL_LABELS.items():
        if label == model_label:
            return get_model_sort_key(model_id)
    return 9999


def load_scale_questions(scale_id: str) -> dict[int, str]:
    """Load questions for a scale from prompt files.

    Returns dict mapping item number to question text.
    """
    import re

    # Try different prompt directories
    for prompt_dir in ["tyler_v2", "tyler", "ai_legitimacy"]:
        prompt_path = PROMPTS_DIR / prompt_dir / f"{scale_id}.txt"
        if prompt_path.exists():
            content = prompt_path.read_text()

            # Extract questions section
            questions_match = re.search(r"<questions>(.*?)</questions>", content, re.DOTALL)
            if questions_match:
                questions_text = questions_match.group(1).strip()
                questions = {}
                for line in questions_text.split("\n"):
                    line = line.strip()
                    # Match pattern like "1. Question text"
                    match = re.match(r"(\d+)\.\s+(.+)", line)
                    if match:
                        item_num = int(match.group(1))
                        question_text = match.group(2)
                        questions[item_num] = question_text
                return questions
    return {}


def load_scale_answers(scale_id: str) -> dict[int, str]:
    """Load answer key for a scale from prompt files.

    Returns dict mapping response number to label text.
    """
    import re

    # Try different prompt directories
    for prompt_dir in ["tyler_v2", "tyler", "ai_legitimacy"]:
        prompt_path = PROMPTS_DIR / prompt_dir / f"{scale_id}.txt"
        if prompt_path.exists():
            content = prompt_path.read_text()

            # Extract answers section
            answers_match = re.search(r"<answers>(.*?)</answer", content, re.DOTALL)
            if answers_match:
                answers_text = answers_match.group(1).strip()
                answers = {}
                for line in answers_text.split("\n"):
                    line = line.strip()
                    # Match pattern like "1 = Agree" or "4 = Strongly Agree"
                    match = re.match(r"(\d+)\s*=\s*(.+)", line)
                    if match:
                        code = int(match.group(1))
                        label = match.group(2).strip()
                        answers[code] = label
                return answers
    return {}


# Scale name to display label mapping
SCALE_LABELS = {
    "compliance": "Compliance",
    "deterrence": "Deterrence",
    "morality": "Morality",
    "obligation": "Obligation",
    "peers": "Peers",
    "performance_courts": "Performance (Courts)",
    "performance_police": "Performance (Police)",
    "support_courts": "Support (Courts)",
    "support_police": "Support (Police)",
}


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrapped confidence interval for the mean."""
    if len(data) < 2:
        mean = np.mean(data)
        return (mean, mean)

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)
    return (lower, upper)


def get_scale_label(scale_id: str) -> str:
    """Convert scale ID to display label."""
    if scale_id in SCALE_LABELS:
        return SCALE_LABELS[scale_id]
    # Fallback: title case with underscores replaced
    return scale_id.replace("_", " ").title()


def get_model_label(model_id: str) -> str:
    """Convert model ID to display label."""
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    # Fallback: clean up the ID for display
    label = model_id.replace("_", " ").replace("-", " ")
    # Remove date suffixes like "2025 04 14"
    parts = label.split()
    cleaned = []
    for part in parts:
        # Skip parts that look like dates (4 digits or 2 digits)
        if part.isdigit() and len(part) in (2, 4):
            continue
        cleaned.append(part)
    return " ".join(cleaned).title() if cleaned else model_id

# Page config - light mode only
st.set_page_config(
    page_title="Legal Attitudes Dashboard",
    page_icon="scales",
    layout="wide",
)

# Custom CSS for Computer Modern-style font and light mode
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=CMU+Serif:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Serif Pro', 'CMU Serif', 'Computer Modern', Georgia, serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Source Serif Pro', 'CMU Serif', 'Computer Modern', Georgia, serif;
    }

    .stMarkdown, .stText, p, span, div {
        font-family: 'Source Serif Pro', 'CMU Serif', 'Computer Modern', Georgia, serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Version string to bust cache when labels change
_LABEL_VERSION = "v2"  # Increment when MODEL_LABELS or SCALE_LABELS change


@st.cache_data
def load_data(file_path: str, _version: str = _LABEL_VERSION) -> pd.DataFrame:
    """Load and cache CSV data."""
    df = pd.read_csv(file_path)
    # Ensure response column is numeric
    if "response" in df.columns:
        df["response"] = pd.to_numeric(df["response"], errors="coerce")
    # Add display label for models
    if "model" in df.columns:
        df["model_label"] = df["model"].apply(get_model_label)
    # Add display label for scales
    if "scale" in df.columns:
        df["scale_label"] = df["scale"].apply(get_scale_label)
    return df


def get_available_datasets() -> dict[str, Path]:
    """Get all available CSV files in the data directory."""
    return {f.stem: f for f in sorted(DATA_DIR.glob("*.csv"))}


def main():
    # Sidebar: Dataset selection
    st.sidebar.header("Data Selection")
    datasets = get_available_datasets()

    if not datasets:
        st.error(f"No CSV files found in {DATA_DIR}")
        return

    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=list(datasets.keys()),
        index=list(datasets.keys()).index("exp1.1.6_tyler_t=1_oai")
        if "exp1.1.6_tyler_t=1_oai" in datasets
        else 0,
    )

    # Load data
    df = load_data(datasets[selected_dataset])

    # Dataset Details
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; margin-bottom: 10px;'><strong>Dataset Details</strong></div>
        <table style='width: 100%; border-collapse: collapse;'>
            <tr style='border-bottom: 1px solid #ddd;'>
                <td style='padding: 6px;'>Models</td>
                <td style='padding: 6px; text-align: right;'>{}</td>
            </tr>
            <tr style='border-bottom: 1px solid #ddd;'>
                <td style='padding: 6px;'>Scales</td>
                <td style='padding: 6px; text-align: right;'>{}</td>
            </tr>
            <tr>
                <td style='padding: 6px;'>Rows</td>
                <td style='padding: 6px; text-align: right;'>{:,}</td>
            </tr>
        </table>
        """.format(df["model"].nunique(), df["scale"].nunique(), len(df)),
        unsafe_allow_html=True,
    )

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")

    # Model filter (show labels, store IDs, sorted by capability)
    models = sorted(df["model"].unique(), key=get_model_sort_key)
    model_to_label = {m: get_model_label(m) for m in models}

    st.sidebar.markdown("#### Models")

    # Initialize or reset session state for models (filter to valid options)
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = models
    else:
        # Filter to only models that exist in current dataset
        valid_models = [m for m in st.session_state["selected_models"] if m in models]
        # If none are valid, select all
        if not valid_models:
            valid_models = models
        st.session_state["selected_models"] = valid_models

    col1, col2 = st.sidebar.columns(2)
    if col1.button("All", key="models_all", width="stretch"):
        st.session_state["selected_models"] = models
    if col2.button("None", key="models_none", width="stretch"):
        st.session_state["selected_models"] = []

    # Provider-specific selection buttons
    openai_models = [m for m in models if m.startswith("gpt-")]
    anthropic_models = [m for m in models if m.startswith("claude-")]
    google_models = [m for m in models if m.startswith("gemini-")]

    # Only show provider buttons if those models exist in dataset
    provider_cols = []
    if openai_models:
        provider_cols.append(("OpenAI", openai_models))
    if anthropic_models:
        provider_cols.append(("Anthropic", anthropic_models))
    if google_models:
        provider_cols.append(("Google", google_models))

    if provider_cols:
        cols = st.sidebar.columns(len(provider_cols))
        for i, (provider_name, provider_models) in enumerate(provider_cols):
            if cols[i].button(provider_name, key=f"models_{provider_name.lower()}", width="stretch"):
                st.session_state["selected_models"] = provider_models

    selected_models = st.sidebar.multiselect(
        "Models",
        options=models,
        default=st.session_state["selected_models"],
        format_func=lambda x: model_to_label.get(x, x),
        label_visibility="collapsed",
    )
    st.session_state["selected_models"] = selected_models

    st.sidebar.markdown("---")

    # Scale filter (show labels, store IDs)
    scales = sorted(df["scale"].unique())
    scale_to_label = {s: get_scale_label(s) for s in scales}

    st.sidebar.markdown("#### Scales")

    # Initialize or reset session state for scales (filter to valid options)
    if "selected_scales" not in st.session_state:
        st.session_state["selected_scales"] = scales
    else:
        # Filter to only scales that exist in current dataset
        valid_scales = [s for s in st.session_state["selected_scales"] if s in scales]
        # If none are valid, select all
        if not valid_scales:
            valid_scales = scales
        st.session_state["selected_scales"] = valid_scales

    col1, col2 = st.sidebar.columns(2)
    if col1.button("All", key="scales_all", width="stretch"):
        st.session_state["selected_scales"] = scales
    if col2.button("None", key="scales_none", width="stretch"):
        st.session_state["selected_scales"] = []

    selected_scales = st.sidebar.multiselect(
        "Scales",
        options=scales,
        default=st.session_state["selected_scales"],
        format_func=lambda x: scale_to_label.get(x, x),
        label_visibility="collapsed",
    )
    st.session_state["selected_scales"] = selected_scales

    # Apply filters
    filtered_df = df[
        (df["model"].isin(selected_models)) & (df["scale"].isin(selected_scales))
    ]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Main content tabs - Response Summary first
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Response Summary", "Item Summary", "Raw Response Viewer", "Refusal Rate Tracking"]
    )

    # Tab 1: Response Summary (now first)
    with tab1:
        # Exclude refusals and NaN values from mean calculation
        non_refusal_df = filtered_df[
            (filtered_df["response"] != REFUSAL_CODE) &
            (filtered_df["response"].notna())
        ]

        if not non_refusal_df.empty:
            # Mean Response Heatmap
            st.markdown("<h3 style='text-align: center;'>Mean Response by Model and Scale</h3>", unsafe_allow_html=True)

            summary = non_refusal_df.groupby(["model_label", "scale_label"])["response"].agg(
                ["mean", "std"]
            )
            pivot_mean = summary["mean"].unstack(level="scale_label").round(2)
            # Sort by model capability (weaker to stronger)
            sorted_index = sorted(pivot_mean.index, key=get_model_label_sort_key)
            pivot_mean = pivot_mean.reindex(sorted_index)
            pivot_mean.index.name = "Model"

            st.dataframe(
                pivot_mean.style.format("{:.2f}").background_gradient(
                    cmap="RdYlGn", axis=None
                ),
                use_container_width=True,
                hide_index=False,
                height=(len(pivot_mean) + 1) * 35 + 3,  # Auto-fit height to rows
                column_config={
                    "_index": st.column_config.TextColumn("Model", width=180),
                },
            )

            # Bar plot: Mean response per model for selected scale
            st.markdown("<h3 style='text-align: center;'>Mean Response by Model and Scale (Bar Plot)</h3>", unsafe_allow_html=True)

            # Scale selector - full width to avoid wrapping
            scale_labels = [scale_to_label.get(s, s) for s in selected_scales]
            selected_scale_label = st.segmented_control(
                "Select Scale",
                options=scale_labels,
                key="scale_for_bar_plot",
                default=scale_labels[0] if scale_labels else None,
            )
            # Map back to scale ID
            label_to_scale = {scale_to_label.get(s, s): s for s in selected_scales}
            selected_scale_for_plot = label_to_scale.get(selected_scale_label, selected_scales[0] if selected_scales else None)

            # Get ALL data for selected scale (including refusals)
            all_scale_data = filtered_df[
                filtered_df["scale"] == selected_scale_for_plot
            ]
            # Get non-refusal data for selected scale
            scale_data = non_refusal_df[
                non_refusal_df["scale"] == selected_scale_for_plot
            ]

            if not all_scale_data.empty:
                # Build label_to_model mapping from ALL selected models
                label_to_model = {get_model_label(m): m for m in selected_models}

                # Get all model labels sorted by capability
                all_model_labels = sorted(
                    [get_model_label(m) for m in selected_models],
                    key=get_model_label_sort_key
                )

                # Calculate mean and 95% bootstrapped CI per model (for non-refusal data)
                model_stats_data = []
                refusal_models = []  # Models with all refusals

                for model_label in all_model_labels:
                    model_id = label_to_model.get(model_label, "")
                    model_data = scale_data[scale_data["model_label"] == model_label]["response"]

                    if len(model_data) == 0:
                        # All refusals for this model
                        refusal_models.append(model_label)
                        model_stats_data.append({
                            "model_label": model_label,
                            "mean": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "is_refusal": True,
                        })
                    else:
                        mean = model_data.mean()
                        lower, upper = bootstrap_ci(model_data.values)
                        model_stats_data.append({
                            "model_label": model_label,
                            "mean": mean,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "is_refusal": False,
                        })

                model_stats = pd.DataFrame(model_stats_data).set_index("model_label")

                # Get model IDs for colors
                model_ids = [label_to_model.get(label, "") for label in model_stats.index]
                bar_colors = [get_model_color(m) for m in model_ids]

                # Calculate x positions with spacing between providers
                x_positions = []
                current_x = 0
                prev_provider = None
                provider_spacing = 0.5  # Extra space between providers

                for label in model_stats.index:
                    model_id = label_to_model.get(label, "")
                    provider = get_provider_from_model(model_id)
                    if prev_provider is not None and provider != prev_provider:
                        current_x += provider_spacing
                    x_positions.append(current_x)
                    current_x += 1
                    prev_provider = provider

                # Create matplotlib figure with serif font
                plt.rcParams["font.family"] = "serif"
                plt.rcParams["font.serif"] = [
                    "CMU Serif",
                    "Computer Modern Roman",
                    "DejaVu Serif",
                    "Georgia",
                    "Times New Roman",
                ]

                fig, ax = plt.subplots(figsize=(10, 4))

                # Plot bars only for non-refusal models
                non_refusal_mask = ~model_stats["is_refusal"]
                if non_refusal_mask.any():
                    non_refusal_stats = model_stats[non_refusal_mask]
                    non_refusal_x = [x_positions[i] for i, is_ref in enumerate(model_stats["is_refusal"]) if not is_ref]
                    non_refusal_colors = [bar_colors[i] for i, is_ref in enumerate(model_stats["is_refusal"]) if not is_ref]

                    yerr_lower = non_refusal_stats["mean"] - non_refusal_stats["ci_lower"]
                    yerr_upper = non_refusal_stats["ci_upper"] - non_refusal_stats["mean"]

                    bars = ax.bar(
                        non_refusal_x,
                        non_refusal_stats["mean"],
                        yerr=[yerr_lower, yerr_upper],
                        capsize=3,
                        color=non_refusal_colors,
                        edgecolor="black",
                        linewidth=0.5,
                        error_kw={"elinewidth": 1, "capthick": 1},
                        zorder=3,
                    )

                # Add "Refusal" labels for refusal models
                for i, (label, row) in enumerate(model_stats.iterrows()):
                    if row["is_refusal"]:
                        ax.text(
                            x_positions[i], 0.15, "Refusal",
                            rotation=90, ha="center", va="bottom",
                            fontsize=15, color="black",
                        )

                # Add y-axis grid lines underneath
                ax.yaxis.grid(True, linestyle="-", alpha=0.3, zorder=0)
                ax.set_axisbelow(True)

                ax.set_xticks(x_positions)
                ax.set_xticklabels(model_stats.index, rotation=30, ha="right")
                ax.set_ylabel("Mean Response")
                ax.set_xlabel("Model")
                ax.set_title(f"{scale_to_label.get(selected_scale_for_plot, selected_scale_for_plot)}", fontsize=15, pad=25)

                # Add provider legend above plot, centered
                providers_in_plot = []
                seen_providers = set()
                for label in model_stats.index:
                    model_id = label_to_model.get(label, "")
                    provider = get_provider_from_model(model_id)
                    if provider not in seen_providers:
                        seen_providers.add(provider)
                        providers_in_plot.append(provider)

                # Create legend patches
                from matplotlib.patches import Patch
                legend_patches = [
                    Patch(facecolor=PROVIDER_COLORS.get(p, "#888888"), edgecolor="black", linewidth=0.5, label=p.title())
                    for p in providers_in_plot
                ]
                ax.legend(
                    handles=legend_patches,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=len(providers_in_plot),
                    frameon=False,
                    fontsize=12,
                )

                # Scale-specific y-axis limits
                scale_ylims = {
                    "compliance": 4.35,
                    "deterrence": 4.35,
                    "morality": 4.35,
                    "obligation": 3.10,
                    "peers": 4.35,
                    "performance_courts": 5.10,
                    "performance_police": 5.10,
                    "support_courts": 4.35,
                    "support_police": 4.35,
                }
                y_max = scale_ylims.get(selected_scale_for_plot, 4.25)
                ax.set_ylim(0, y_max)

                # Set xlim with padding for edge labels (especially refusal text)
                x_padding = 0.6
                ax.set_xlim(min(x_positions) - x_padding, max(x_positions) + x_padding)

                # Add scale-specific labels
                if selected_scale_for_plot == "compliance":
                    ax.text(-0.07, 1.0, "Full\nCompliance", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "No\nCompliance", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "deterrence":
                    ax.text(-0.07, 1.0, "Greater\nDeterrence\nEffect", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "No\nDeterrence\nEffect", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "morality":
                    ax.text(-0.07, 1.0, "Morally\nPermissive", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "Morally\nStrict", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "obligation":
                    ax.text(-0.07, 1.0, "Less\nObligation", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "More\nObligation", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "peers":
                    ax.text(-0.07, 1.0, "Weaker\nPeer Effects", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "Greater\nPeer Effects", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "performance_courts":
                    ax.text(-0.07, 1.0, "Worse\nPerformance", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "Better\nPerformance", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "performance_police":
                    ax.text(-0.07, 1.0, "Worse\nPerformance", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "Better\nPerformance", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "support_courts":
                    ax.text(-0.07, 1.0, "Less\nSupport", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "More\nSupport", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                elif selected_scale_for_plot == "support_police":
                    ax.text(-0.07, 1.0, "Less\nSupport", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                    ax.text(-0.07, 0.0, "More\nSupport", transform=ax.transAxes,
                            fontsize=9, va="center", ha="center")
                # Add value labels on bars (skip refusal models)
                for i, (x_pos, mean, upper, is_ref) in enumerate(
                    zip(x_positions, model_stats["mean"], model_stats["ci_upper"], model_stats["is_refusal"])
                ):
                    if not is_ref:
                        ax.annotate(
                            f"{mean:.2f}",
                            xy=(x_pos, upper + 0.1),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                # Adjust layout to make room for legend above
                plt.tight_layout()
                fig.subplots_adjust(top=0.85)

                # Center the plot at 90% width to match selector
                col_left, col_center, col_right = st.columns([0.05, 0.9, 0.05])
                with col_center:
                    st.pyplot(fig)
                plt.close()
            else:
                st.info(f"No data available for scale: {selected_scale_label}")

        else:
            st.info("All responses are refusals - no summary statistics available.")

    # Tab 2: Item Summary
    with tab2:
        st.markdown("<h3 style='text-align: center;'>Item Response Distribution</h3>", unsafe_allow_html=True)

        # Dropdown for scale only (models come from sidebar filter)
        item_summary_scale = st.selectbox(
            "Scale",
            options=selected_scales,
            key="item_summary_scale",
            format_func=lambda x: scale_to_label.get(x, x),
        )

        # Get ALL data for selected scale and all selected models
        all_item_data = filtered_df[
            filtered_df["scale"] == item_summary_scale
        ]

        if not all_item_data.empty and len(selected_models) > 0:
            # Load questions and answers for this scale
            questions = load_scale_questions(item_summary_scale)
            answers = load_scale_answers(item_summary_scale)

            # Get unique items sorted (filter out non-numeric items like "error")
            def is_numeric(val):
                try:
                    int(val)
                    return True
                except (ValueError, TypeError):
                    return False

            items = sorted([i for i in all_item_data["item"].unique() if is_numeric(i)], key=lambda x: int(x))

            # Get response codes from the data itself (handles scales with varying answer codes)
            # Get all unique non-null responses, excluding refusal code
            data_response_codes = sorted([
                int(r) for r in all_item_data["response"].dropna().unique()
                if is_numeric(r) and int(r) != REFUSAL_CODE
            ])
            # Use answers from prompt if available, otherwise use data
            if answers:
                all_response_codes = sorted(answers.keys()) + [REFUSAL_CODE]
            else:
                all_response_codes = data_response_codes + [REFUSAL_CODE]

            # Create response labels for x-axis (including refusal)
            response_labels = {code: f"{code} = {label}" for code, label in answers.items()}
            # For codes not in answers dict, just show the number
            for code in all_response_codes:
                if code not in response_labels:
                    if code == REFUSAL_CODE:
                        response_labels[code] = f"{code} = Refusal"
                    else:
                        response_labels[code] = str(code)

            # Get models sorted by capability
            models_sorted = sorted(selected_models, key=get_model_sort_key)

            # Determine providers and build color scheme
            providers_present = list(dict.fromkeys([get_provider_from_model(m) for m in models_sorted]))
            multiple_providers = len(providers_present) > 1

            # Build model colors based on provider logic
            def hex_to_rgba(hex_color, alpha=1.0):
                """Convert hex color to RGBA tuple."""
                hex_color = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
                return (r, g, b, alpha)

            if multiple_providers:
                # Group models by provider and assign opacity levels
                models_by_provider = {}
                for model in models_sorted:
                    provider = get_provider_from_model(model)
                    if provider not in models_by_provider:
                        models_by_provider[provider] = []
                    models_by_provider[provider].append(model)

                # Assign colors with varying opacity per provider
                model_colors_dict = {}
                for provider, provider_models in models_by_provider.items():
                    base_color = PROVIDER_COLORS.get(provider, "#888888")
                    n_models_in_provider = len(provider_models)
                    # Opacity from 0.4 to 1.0
                    opacities = np.linspace(0.4, 1.0, n_models_in_provider)
                    for i, model in enumerate(provider_models):
                        model_colors_dict[model] = hex_to_rgba(base_color, opacities[i])

                model_colors = [model_colors_dict[m] for m in models_sorted]
            else:
                # Single provider: use normal color wheel
                tab10_colors = plt.cm.tab10(np.linspace(0, 1, len(models_sorted)))
                model_colors = [tuple(c) for c in tab10_colors]

            # Set up matplotlib style
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = [
                "CMU Serif",
                "Computer Modern Roman",
                "DejaVu Serif",
                "Georgia",
                "Times New Roman",
            ]

            # Create a plot for each item
            for item in items:
                question_text = questions.get(int(item), f"Item {item}")

                # Centered item text above the plot
                st.markdown(
                    f"<div style='text-align: center;'><strong style='font-size: 1.3em;'>Item {int(item)}</strong><br>"
                    f"<span style='font-size: 1.2em;'>{question_text}</span></div>",
                    unsafe_allow_html=True,
                )

                fig, ax = plt.subplots(figsize=(9, 2.5))

                n_models = len(models_sorted)
                n_responses = len(all_response_codes)
                bar_width = 0.8 / n_models
                x = np.arange(n_responses)

                # Plot bars for each model
                for i, model in enumerate(models_sorted):
                    model_item_data = all_item_data[
                        (all_item_data["model"] == model) &
                        (all_item_data["item"] == item)
                    ]["response"]

                    # Count responses and convert to frequencies
                    response_counts = model_item_data.value_counts().reindex(all_response_codes, fill_value=0)
                    total = response_counts.sum()
                    response_freq = response_counts / total if total > 0 else response_counts

                    offset = (i - n_models / 2 + 0.5) * bar_width
                    bars = ax.bar(
                        x + offset,
                        response_freq.values,
                        bar_width,
                        label=model_to_label.get(model, model),
                        color=model_colors[i],
                        edgecolor="black",
                        linewidth=0.3,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels([response_labels.get(c, str(c)) for c in all_response_codes], fontsize=8)
                ax.set_ylabel("Freq", fontsize=9)
                ax.set_ylim(0, 1.15)
                ax.yaxis.grid(True, linestyle="-", alpha=0.3, zorder=0)
                ax.set_axisbelow(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Centered legend below the plot as a grid, organized by provider rows
                if multiple_providers:
                    # Find max models per provider for grid columns
                    max_models_per_provider = max(
                        len([m for m in models_sorted if get_provider_from_model(m) == p])
                        for p in providers_present
                    )
                    legend_html = f"<div style='display: grid; grid-template-columns: repeat({max_models_per_provider}, 1fr); gap: 8px 20px; justify-items: center; margin: 10px auto; max-width: 90%;'>"
                    for provider in providers_present:
                        provider_models = [m for m in models_sorted if get_provider_from_model(m) == provider]
                        for model in provider_models:
                            color = model_colors_dict[model]
                            color_css = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"
                            label = model_to_label.get(model, model)
                            legend_html += f"<div style='display: flex; align-items: center;'>"
                            legend_html += f"<span style='background-color: {color_css}; border: 1px solid black; display: inline-block; width: 16px; height: 16px; margin-right: 5px; flex-shrink: 0;'></span>"
                            legend_html += f"<span style='white-space: nowrap;'>{label}</span></div>"
                        # Add empty cells to fill the row
                        for _ in range(max_models_per_provider - len(provider_models)):
                            legend_html += "<div></div>"
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)
                else:
                    # Single provider: grid layout
                    n_cols = min(len(models_sorted), 5)  # Max 5 per row
                    legend_html = f"<div style='display: grid; grid-template-columns: repeat({n_cols}, 1fr); gap: 8px 20px; justify-items: center; margin: 10px auto; max-width: 90%;'>"
                    for i, model in enumerate(models_sorted):
                        color = model_colors[i]
                        color_hex = "#{:02x}{:02x}{:02x}".format(
                            int(color[0] * 255),
                            int(color[1] * 255),
                            int(color[2] * 255),
                        )
                        label = model_to_label.get(model, model)
                        legend_html += f"<div style='display: flex; align-items: center;'>"
                        legend_html += f"<span style='background-color: {color_hex}; border: 1px solid black; display: inline-block; width: 16px; height: 16px; margin-right: 5px; flex-shrink: 0;'></span>"
                        legend_html += f"<span style='white-space: nowrap;'>{label}</span></div>"
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)

                st.markdown("---")
        else:
            st.info("No data available for the selected scale. Make sure models are selected in the sidebar.")

    # Tab 3: Raw Response Viewer
    with tab3:
        st.header("Raw Response Viewer")
        st.markdown("Browse individual responses with filtering options.")

        col1, col2, col3 = st.columns(3)
        with col1:
            viewer_model = st.selectbox(
                "Model",
                options=selected_models,
                key="viewer_model",
                format_func=lambda x: model_to_label.get(x, x),
            )
        with col2:
            viewer_scale = st.selectbox(
                "Scale",
                options=selected_scales,
                key="viewer_scale",
                format_func=lambda x: scale_to_label.get(x, x),
            )
        with col3:
            available_repeats = sorted(
                filtered_df[
                    (filtered_df["model"] == viewer_model)
                    & (filtered_df["scale"] == viewer_scale)
                ]["repeat"].unique()
            )
            viewer_repeat = st.selectbox(
                "Repeat",
                options=available_repeats if available_repeats else [0],
                key="viewer_repeat",
            )

        # Get responses for selected combination
        viewer_df = filtered_df[
            (filtered_df["model"] == viewer_model)
            & (filtered_df["scale"] == viewer_scale)
            & (filtered_df["repeat"] == viewer_repeat)
        ].sort_values("item")

        if not viewer_df.empty:
            # Load questions and answers for this scale
            questions = load_scale_questions(viewer_scale)
            answers = load_scale_answers(viewer_scale)

            # Calculate average response across all repeats for this model/scale
            all_repeats_df = filtered_df[
                (filtered_df["model"] == viewer_model)
                & (filtered_df["scale"] == viewer_scale)
                & (filtered_df["response"] != REFUSAL_CODE)  # Exclude refusals from average
            ]
            avg_by_item = all_repeats_df.groupby("item")["response"].mean()

            # Format response as "4 = Strongly Agree"
            def format_response(resp):
                try:
                    resp_int = int(resp)
                    if resp_int == REFUSAL_CODE:
                        return f"{resp_int}\n(Refusal/Missing)"
                    label = answers.get(resp_int, "")
                    if label:
                        return f"{resp_int}\n({label})"
                    return str(resp_int)
                except (ValueError, TypeError):
                    return str(resp)

            # Check if raw_output column exists
            has_raw_output = "raw_output" in viewer_df.columns

            # Display as a clean table with question text
            display_df = viewer_df[["item", "response"]].copy()

            def get_question(item_val):
                try:
                    return questions.get(int(item_val), "")
                except (ValueError, TypeError):
                    return ""

            display_df["question"] = display_df["item"].apply(get_question)
            display_df["response_formatted"] = display_df["response"].apply(format_response)
            display_df["avg_response"] = display_df["item"].apply(
                lambda x: avg_by_item.get(x, np.nan)
            )

            # Use HTML table for better text wrapping
            html_rows = []
            for idx, row in display_df.iterrows():
                avg_val = row['avg_response']
                avg_str = f"{avg_val:.2f}" if not np.isnan(avg_val) else "N/A"

                html_rows.append(
                    f"<tr><td style='width: 50px; padding: 8px; border-bottom: 1px solid #eee;'>{row['item']}</td>"
                    f"<td style='white-space: normal; word-wrap: break-word; padding: 8px; border-bottom: 1px solid #eee;'>{row['question']}</td>"
                    f"<td style='width: 150px; padding: 8px; border-bottom: 1px solid #eee;'>{row['response_formatted']}</td>"
                    f"<td style='width: 80px; padding: 8px; border-bottom: 1px solid #eee; text-align: center;'>{avg_str}</td></tr>"
                )

            html_table = f"""
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>
                    <tr style='border-bottom: 2px solid #ddd;'>
                        <th style='text-align: left; padding: 8px;'>Item</th>
                        <th style='text-align: left; padding: 8px;'>Question</th>
                        <th style='text-align: left; padding: 8px;'>Response</th>
                        <th style='text-align: center; padding: 8px;'>Avg (Non-refusal)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(html_rows)}
                </tbody>
            </table>
            """
            st.markdown(html_table, unsafe_allow_html=True)

            # Display raw output below table (one per scale)
            if has_raw_output:
                raw_output = viewer_df["raw_output"].iloc[0]
                if pd.notna(raw_output) and str(raw_output).strip():
                    st.markdown("---")
                    st.markdown("**Raw Model Output**")
                    st.code(str(raw_output), language=None)

    # Tab 4: Refusal Rate Tracking
    with tab4:
        # Refusal Fraction Heatmap
        st.markdown("<h3 style='text-align: center;'>Refusal Fraction by Model and Scale</h3>", unsafe_allow_html=True)
        refusal_frac = (
            filtered_df.groupby(["model_label", "scale_label"])["response"]
            .apply(lambda x: (x == REFUSAL_CODE).mean())
            .unstack(level="scale_label")
            .round(2)
        )
        # Sort by model capability (weaker to stronger)
        sorted_index = sorted(refusal_frac.index, key=get_model_label_sort_key)
        refusal_frac = refusal_frac.reindex(sorted_index)
        refusal_frac.index.name = "Model"

        st.dataframe(
            refusal_frac.style.format("{:.2f}").background_gradient(
                cmap="Reds", axis=None, vmin=0, vmax=1
            ),
            use_container_width=True,
            hide_index=False,
            height=(len(refusal_frac) + 1) * 35 + 3,
            column_config={
                "_index": st.column_config.TextColumn("Model", width=180),
            },
        )

        # Refusal Count Heatmap
        st.markdown("<h3 style='text-align: center;'>Refusal Count by Model and Scale</h3>", unsafe_allow_html=True)
        refusal_count = (
            filtered_df.groupby(["model_label", "scale_label"])["response"]
            .apply(lambda x: (x == REFUSAL_CODE).sum())
            .unstack(level="scale_label")
        )
        # Sort by model capability (weaker to stronger)
        sorted_index = sorted(refusal_count.index, key=get_model_label_sort_key)
        refusal_count = refusal_count.reindex(sorted_index)

        # Calculate total queries per model/scale (repeats x items)
        total_queries = (
            filtered_df.groupby(["model_label", "scale_label"])
            .size()
            .unstack(level="scale_label")
        )
        total_queries = total_queries.reindex(sorted_index)

        # Add total queries row
        refusal_count_with_total = refusal_count.copy()
        refusal_count_with_total.loc["Total Queries"] = total_queries.iloc[0]  # Same for all models
        refusal_count_with_total.index.name = "Model"

        st.dataframe(
            refusal_count_with_total.style.format("{:.0f}").background_gradient(
                cmap="Reds", axis=None, subset=pd.IndexSlice[refusal_count.index, :]
            ),
            use_container_width=True,
            hide_index=False,
            height=(len(refusal_count_with_total) + 1) * 35 + 3,
            column_config={
                "_index": st.column_config.TextColumn("Model", width=180),
            },
        )


if __name__ == "__main__":
    main()
