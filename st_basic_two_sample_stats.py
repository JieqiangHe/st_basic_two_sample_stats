import streamlit as st
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

def parse_input_data():
    try:
        # Create an empty dataframe with two columns and 5 rows
        df = pd.DataFrame(columns=['Group1', 'Group2'], index=range(5))

        # Display editable dataframe
        edited_df = st.data_editor(df, num_rows="dynamic")

        if len(edited_df.columns) < 2:
            st.error("Please provide at least two columns of data")
            return None, None

        # Convert to numeric and drop non-numeric values
        group1_data = pd.to_numeric(edited_df.iloc[:, 0], errors='coerce').dropna().tolist()
        group2_data = pd.to_numeric(edited_df.iloc[:, 1], errors='coerce').dropna().tolist()

        if len(group1_data) == 0 or len(group2_data) == 0:
            st.error("Please provide valid numeric data in both columns")
            return None, None

        return group1_data, group2_data
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def perform_test(group1, group2, test_type):
    # Convert to numpy arrays and ensure numeric values
    group1 = np.array(group1, dtype=float)
    group2 = np.array(group2, dtype=float)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        st.error("Invalid data - no valid numeric values found")
        return None

    if test_type == "Unpaired two-sample t-test (parametric)":
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        df = len(group1) + len(group2) - 2
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + (len(group2)-1)*np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
        ci = stats.t.interval(0.95, df, loc=np.mean(group1)-np.mean(group2),
                            scale=pooled_std * np.sqrt(1/len(group1) + 1/len(group2)))
        return {
            'test_type': "Independent two-sample t-test (equal variances)",
            'p_value': p_value,
            'confidence_interval': ci,
            'degrees_of_freedom': df,
            'statistic': t_stat
        }
    else:
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        # Calculate rank-biserial correlation (effect size)
        combined = np.concatenate([group1, group2])
        ranks = stats.rankdata(combined)
        rank1 = ranks[:len(group1)]
        rank2 = ranks[len(group1):]
        mean_rank1 = np.mean(rank1)
        mean_rank2 = np.mean(rank2)
        r_biserial = 1 - (2 * u_stat) / (len(group1) * len(group2))

        return {
            'test_type': "Wilcoxon rank-sum test (Mann-Whitney U) (unpaired, non-Gaussian)",
            'p_value': p_value,
            'statistic': u_stat,
            'effect_size': r_biserial,
            'mean_rank_group1': mean_rank1,
            'mean_rank_group2': mean_rank2
        }

def create_plots(group1_data, group2_data, group1_name, group2_name, palette_name):
    # Set Arial font and style
    rcParams['font.family'] = 'Arial'
    sns.set_style("ticks")

    # Combine data for plotting
    combined_data = pd.concat([
        pd.DataFrame({'Value': group1_data, 'Group': group1_name}),
        pd.DataFrame({'Value': group2_data, 'Group': group2_name})
    ])

    # Create individual figures for each plot type
    figures = []

    # Scatter plot with bar (mean ± SD)
    fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=600)
    sns.barplot(data=combined_data, x='Group', y='Value', ci='sd', capsize=0.1, ax=ax1, palette=palette_name)
    sns.stripplot(data=combined_data, x='Group', y='Value', color='black', ax=ax1)
    ax1.set_xlabel('')
    figures.append(fig1)
    plt.close(fig1)

    # Boxplot with all points
    fig2, ax2 = plt.subplots(figsize=(4,4), dpi=600)
    sns.boxplot(data=combined_data, x='Group', y='Value', ax=ax2, palette=palette_name)
    sns.stripplot(data=combined_data, x='Group', y='Value', color='black', ax=ax2)
    ax2.set_xlabel('')
    figures.append(fig2)
    plt.close(fig2)

    # Violin plot with boxplot inside and all points
    fig3, ax3 = plt.subplots(figsize=(4, 4), dpi=600)
    sns.violinplot(data=combined_data, x='Group', y='Value', hue='Group', ax=ax3, palette=palette_name, legend=False, inner='box')
    sns.stripplot(data=combined_data, x='Group', y='Value', color='black', ax=ax3)
    ax3.set_xlabel('')
    figures.append(fig3)
    plt.close(fig3)

    return figures

def save_plots_to_pdf(figures):
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight', dpi=600)
    pdf_buffer.seek(0)
    return pdf_buffer

def main():
    st.markdown("""
    <h1 style='text-align: center; color: #4a6baf;'>
        Two Sample Statistical Tests: t-test & Mann-Whitney U
    </h1>
    """, unsafe_allow_html=True)

    group1_name = st.text_input("**Group 1 Name:**", "*Group1*")
    group2_name = st.text_input("**Group 2 Name:**", "*Group2*")

    st.markdown("### Data Input", unsafe_allow_html=True)
    st.write("Upload or edit your data below (first column = Group 1, second column = Group 2):")
    group1_data, group2_data = parse_input_data()
    if not group1_data or not group2_data:
        st.stop()

    # Add color palette selection
    palette_options = [
        "deep", "muted", "bright", "pastel", "dark", "colorblind",
        "rocket", "mako", "flare", "crest", "viridis", "plasma",
        "inferno", "magma", "cividis", "coolwarm", "Spectral"
    ]
    palette_name = st.selectbox("Select Color Palette:", palette_options, index=palette_options.index("deep"))

    test_type = st.radio("Test Type:",
                        ("Unpaired two-sample t-test (parametric)",
                         "Wilcoxon rank-sum test (non-parametric/Mann-Whitney U)"))

    # Auto-run the test when data is available
    result = perform_test(group1_data, group2_data, test_type)

    if result is not None:
        st.markdown("### Test Results", unsafe_allow_html=True)
        st.markdown(f"**Test performed:** `{result['test_type']}`")
        st.markdown(f"**Test statistic:** `{result['statistic']:.10f}`")
        st.markdown(f"**p-value:** `{result['p_value']:.10f}`")

        if 'confidence_interval' in result:
            st.markdown(f"**95% Confidence Interval:** `({result['confidence_interval'][0]:.10f}, {result['confidence_interval'][1]:.10f})`")

        if 'degrees_of_freedom' in result:
            st.markdown(f"**Degrees of freedom:** `{result['degrees_of_freedom']}`")

        # Display plots
        st.markdown("### Data Visualization", unsafe_allow_html=True)
        figures = create_plots(group1_data, group2_data, group1_name, group2_name, palette_name)
        for fig in figures:
            st.pyplot(fig)

        # Add PDF download button
        st.markdown("### Export Plots", unsafe_allow_html=True)
        pdf_buffer = save_plots_to_pdf(figures)
        st.download_button(
            label="Download Plots as PDF (for Adobe Illustrator)",
            data=pdf_buffer,
            file_name="statistical_plots.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
