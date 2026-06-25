import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging
from config import call_llm, call_llm_with_cache

logger = logging.getLogger("insights_generator")

#  Light professional chart theme (matches app UI) 
PRIMARY   = "#2563EB"   # electric blue
ACCENT2   = "#7C3AED"   # violet
ACCENT3   = "#10B981"   # emerald
BG_DARK   = "#FFFFFF"
BG_PANEL  = "#a6d1f7"
TEXT_MAIN = "#1E293B"
TEXT_MUTED= "#64748B"
GRID_LINE = "#E2E8F0"

CHART_COLORS = [
    "#2563EB", "#7C3AED", "#10B981", "#F59E0B",
    "#EF4444", "#06B6D4", "#EC4899", "#84CC16",
    "#F97316", "#8B5CF6"
]

import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_DARK,
    "axes.edgecolor":    GRID_LINE,
    "axes.labelcolor":   TEXT_MAIN,
    "axes.titlecolor":   TEXT_MAIN,
    "axes.grid":         True,
    "axes.prop_cycle":   mpl.cycler(color=CHART_COLORS),
    "grid.color":        GRID_LINE,
    "grid.linewidth":    0.7,
    "text.color":        TEXT_MAIN,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  BG_DARK,
    "legend.edgecolor":  GRID_LINE,
    "legend.labelcolor": TEXT_MAIN,
    "figure.dpi":        110,
})
sns.set_style("whitegrid", {
    "axes.facecolor":  BG_DARK,
    "figure.facecolor": BG_DARK,
    "grid.color":      GRID_LINE,
})


def show_centered_plot(fig, use_container_width=True, clear_figure=True, ratio=0.7):
    """Render a Matplotlib figure centered in the Streamlit interface"""
    side_ratio = (1.0 - ratio) / 2.0
    c1, c2, c3 = st.columns([side_ratio, ratio, side_ratio])
    with c2:
        st.pyplot(fig, use_container_width=True, clear_figure=clear_figure)


class DataInsightsGenerator:
    """Generate comprehensive insights from query results"""
    
    def __init__(self, result_df: pd.DataFrame):
        self.df = result_df
        raw_numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        
        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in self.df.columns:
            if col in raw_numeric_cols:
                # Distinguish between value/measure columns and identifier/low-cardinality integer categories (e.g. Chapter 7/11/13, Year 2024)
                is_value_col = any(x in str(col).lower() for x in ['count', 'total', 'frequency', 'value', 'sum', 'amount', 'pct', 'ratio'])
                is_low_cardinality_int = (
                    pd.api.types.is_integer_dtype(self.df[col]) and 
                    self.df[col].nunique() <= 20 and
                    not is_value_col
                )
                if is_low_cardinality_int and len(raw_numeric_cols) > 1:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)
                
        self.date_cols = self._detect_date_columns()
    
    def _detect_date_columns(self) -> List[str]:
        """Detect columns that represent dates"""
        date_cols = []
        for col in self.df.columns:
            if any(token in str(col).lower() for token in ["date", "time", "opened", "filed", "conversion", "status_date"]):
                try:
                    pd.to_datetime(self.df[col], errors="coerce")
                    if pd.to_datetime(self.df[col], errors="coerce").notna().sum() / len(self.df) >= 0.5:
                        date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the dataset"""
        stats = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum()
        }
        return stats
    
    def get_numeric_insights(self) -> Dict:
        """Generate detailed insights for numeric columns"""
        insights = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            insights[col] = {
                "count": len(col_data),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "q25": col_data.quantile(0.25),
                "q75": col_data.quantile(0.75),
                "skewness": col_data.skew(),
                "has_outliers": self._detect_outliers(col_data),
                "outlier_count": self._count_outliers(col_data)
            }
        
        return insights
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Detect if a series has outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).any()
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count number of outliers in a series"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def get_categorical_insights(self) -> Dict:
        """Generate detailed insights for categorical columns"""
        insights = {}
        
        for col in self.categorical_cols:
            col_data = self.df[col].astype(str).value_counts(dropna=True)
            
            insights[col] = {
                "unique_count": self.df[col].nunique(),
                "most_common": col_data.index[0] if len(col_data) > 0 else None,
                "most_common_freq": int(col_data.iloc[0]) if len(col_data) > 0 else 0,
                "least_common": col_data.index[-1] if len(col_data) > 0 else None,
                "least_common_freq": int(col_data.iloc[-1]) if len(col_data) > 0 else 0,
                "top_categories": col_data.head().to_dict(),
                "missing_count": self.df[col].isnull().sum()
            }
        
        return insights
    
    def detect_trends(self) -> Optional[Tuple[str, str]]:
        """Detect if data has date and numeric columns for trend analysis"""
        if self.date_cols and self.numeric_cols:
            return self.date_cols[0], self.numeric_cols[0]
        return None
    
    def get_correlations(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix for numeric columns"""
        if len(self.numeric_cols) >= 2:
            return self.df[self.numeric_cols].corr()
        return None
    
    def get_missing_data_report(self) -> Dict:
        """Generate report on missing data"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        report = {}
        for col in self.df.columns:
            if missing[col] > 0:
                report[col] = {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col])
                }
        
        return report


class InsightVisualizer:
    """Handle visualization of insights with smart recommendations"""
    
    def __init__(self, result_df: pd.DataFrame, insights_gen: DataInsightsGenerator):
        self.df = result_df
        self.insights = insights_gen

    def _aggregate_by_category(self, cat_col: str, num_col: str) -> pd.Series:
        """Helper to aggregate a numeric column by a category using sum or mean depending on the column name."""
        num_lower = str(num_col).lower()
        if any(x in num_lower for x in ["avg", "mean", "score", "pct", "ratio", "percentage"]):
            return self.df.groupby(cat_col)[num_col].mean()
        return self.df.groupby(cat_col)[num_col].sum()
    
    def _is_year_column(self, col_name: str, values: pd.Series) -> bool:
        """Detect whether a categorical column likely represents years"""
        col_lower = str(col_name).lower()
        if "year" in col_lower:
            return True

        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().all() and numeric_values.between(1900, 2100).all():
            return True

        parsed_dates = pd.to_datetime(values, errors="coerce", format="%Y")
        if parsed_dates.notna().all():
            return True

        return False

    def _sort_bar_data_by_category(self, bar_data: pd.DataFrame, cat_col: str, value_col: str) -> pd.DataFrame:
        """Sort chart categories chronologically when they represent years"""
        if self._is_year_column(cat_col, bar_data[cat_col]):
            bar_data = bar_data.copy()
            bar_data["_sort_key"] = pd.to_numeric(bar_data[cat_col], errors="coerce")
            if bar_data["_sort_key"].notna().all():
                return bar_data.sort_values("_sort_key").drop(columns=["_sort_key"])

        return bar_data.sort_values(value_col, ascending=True)

    def _build_llm_summary_prompt(
        self,
        stats: Dict,
        numeric_insights: Dict,
        categorical_insights: Dict,
    ) -> str:
        """Build a prompt for the LLM summarizing dataset trends and drivers."""
        prompt_lines = [
            "Role: You are an elite strategic credit risk advisor and business intelligence analyst.",
            "Task: Generate a highly polished, client-focused 'Trend Summary Analysis' based on the dataset characteristics provided below.",
            "",
            "CRITICAL FORMATTING RULES (follow exactly):",
            "1. Tone: Professional, executive-level, clear, and highly authoritative. Speak directly to business leaders/clients.",
            "2. Use the exact markdown structure shown below — do NOT merge sections into a single paragraph.",
            "3. Each section MUST start on its own new line with a markdown header (## or ###).",
            "4. The 'Key Drivers & Concentrations' section MUST be a bullet list using '- ' prefix for each item.",
            "5. Use **bold** to emphasize key numbers, percentages, and names.",
            "6. Do NOT write prose paragraphs where bullet lists are required.",
            "7. Keep total output under 160 words. No introductory or concluding meta-text.",
            "",
            "REQUIRED OUTPUT STRUCTURE (use this exact format):",
            "## Executive Trend Overview",
            "<1-2 sentence summary of overall volume and distribution trajectory>",
            "",
            "## Key Drivers & Concentrations",
            "- <Driver 1: e.g., top states/territories with % share>",
            "- <Driver 2: e.g., dominant bankruptcy chapter with % share>",
            "- <Driver 3: e.g., primary risk profile or debtor type>",
            "",
            "## Strategic Client Implications",
            "<1-2 sentences on portfolio management, resource allocation, or operational risk>",
            "",
            "Tone: Professional, executive-level, authoritative. Speak directly to business leaders.",
            "Focus: Translate statistics into strategic business insights. Avoid column names, SQL, or data-quality language.",
            "",
            "Dataset Metadata & Aggregated Statistics:",
        ]

        if self.insights.date_cols:
            prompt_lines.append(f"- Temporal/Date Columns detected: {', '.join(self.insights.date_cols)}")

        if self.insights.numeric_cols:
            prompt_lines.append(f"- Numeric Columns detected: {', '.join(self.insights.numeric_cols)}")
            prompt_lines.append("  Summary stats:")
            for col, col_stats in list(numeric_insights.items())[:3]:
                prompt_lines.append(
                    f"    * {col}: Average={round(col_stats['mean'], 2)}, Median={round(col_stats['median'], 2)}, "
                    f"Min={round(col_stats['min'], 2)}, Max={round(col_stats['max'], 2)}, Outliers count={col_stats['outlier_count']}"
                )

        if self.insights.categorical_cols:
            prompt_lines.append(f"- Categorical Columns detected: {', '.join(self.insights.categorical_cols)}")
            prompt_lines.append("  Top distributions:")
            for col, cat_stats in list(categorical_insights.items())[:3]:
                top_categories = cat_stats.get('top_categories', {})
                category_summary = ', '.join([
                    f"{k} (count: {v})" for k, v in list(top_categories.items())[:3]
                ])
                prompt_lines.append(
                    f"    * {col}: Unique count={cat_stats['unique_count']}, Top occurrences={category_summary}"
                )

        prompt_lines.append("")
        prompt_lines.append("Generate the Trend Summary Analysis now, adhering strictly to the Tone & Style Guidelines. Do not include any introductory or concluding meta-text.")

        return "\n".join(prompt_lines)

    def _generate_llm_summary(self, stats: Dict) -> Optional[str]:
        """Generate an LLM-based executive summary of dataset insights."""
        try:
            numeric_insights = self.insights.get_numeric_insights()
            categorical_insights = self.insights.get_categorical_insights()
            prompt = self._build_llm_summary_prompt(stats, numeric_insights, categorical_insights)
            llm_response = call_llm_with_cache(prompt, temperature=0.1)
            return llm_response.strip() if llm_response else None
        except Exception as e:
            logger.warning("LLM summary generation failed: %s", e)
            return None

    def render_executive_summary(self):
        """Render executive summary with key metrics"""
        stats = self.insights.generate_summary_statistics()

        # st.subheader(" Data Summary")

        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("Total Records", stats["total_rows"])
        # with col2:
        #     st.metric("Columns", stats["total_columns"])
        # with col3:
        #     st.metric("Duplicates", stats["duplicate_rows"])
        # with col4:
        #     st.metric("Memory (MB)", stats["memory_usage_mb"])

        llm_summary = self._generate_llm_summary(stats)
        if llm_summary:
            import re
            # Scale heading font sizes to 60% of Streamlit defaults:
            # ## (h2) default ~1.75rem → 1.05rem | ### (h3) default ~1.25rem → 0.75rem
            scaled = re.sub(
                r'^## (.+)$',
                r'<h2 style="font-size:1.05rem;font-weight:700;margin:10px 0 4px 0;">\1</h2>',
                llm_summary, flags=re.MULTILINE
            )
            scaled = re.sub(
                r'^### (.+)$',
                r'<h3 style="font-size:0.75rem;font-weight:600;margin:6px 0 4px 0;">\1</h3>',
                scaled, flags=re.MULTILINE
            )
            st.markdown(scaled, unsafe_allow_html=True)

        # Missing data report
        missing_report = self.insights.get_missing_data_report()
        # if missing_report:
        #     with st.expander(" Missing Data Details", expanded=False):
        #         missing_df = pd.DataFrame([
        #             {"Column": col, "Missing Count": data["count"], "% Missing": data["percentage"]}
        #             for col, data in missing_report.items()
        #         ])
        #         st.dataframe(missing_df, use_container_width=True)
    
    def render_numeric_analysis(self):
        """Render comprehensive numeric column analysis"""
        numeric_insights = self.insights.get_numeric_insights()
        
        if not numeric_insights:
            return
        
        # st.subheader(" Numeric Columns Analysis")
        
        # Summary statistics table
        # with st.expander(" Summary Statistics", expanded=True):
        #     stats_data = []
        #     for col, stats in numeric_insights.items():
        #         stats_data.append({
        #             "Column": col,
        #             "Count": stats["count"],
        #             "Mean": round(stats["mean"], 2),
        #             "Median": round(stats["median"], 2),
        #             "Std Dev": round(stats["std"], 2),
        #             "Min": round(stats["min"], 2),
        #             "Max": round(stats["max"], 2),
        #             "Outliers": stats["outlier_count"]
        #         })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # Visualizations for key numeric columns
        key_numeric = self.insights.numeric_cols[:3]  # Focus on top 3 numeric columns
        
        if len(key_numeric) > 0:
            st.subheader("Numeric Distributions (Bar Plot)")

            num_charts = len(key_numeric)
            # Use full-width single column for 1 chart, else split into columns
            use_full_width = (num_charts == 1)
            if use_full_width:
                chart_containers = [st.container()]
            else:
                chart_containers = st.columns(min(3, num_charts))
            
            for idx, col in enumerate(key_numeric):
                col_data = self.df[col].dropna()
                if len(col_data) < 2:
                    continue

                # ── Adaptive figsize ──────────────────────────────────────────
                is_aggregated = (
                    len(self.insights.categorical_cols) >= 1
                    and len(self.insights.numeric_cols) == 1
                )
                if is_aggregated:
                    cat_col = self.insights.categorical_cols[0]
                    agg_series = self._aggregate_by_category(cat_col, col)
                    bar_data = pd.DataFrame({cat_col: agg_series.index, col: agg_series.values})
                    n_bars = len(bar_data)
                else:
                    n_bars = 15  # histogram bins

                if use_full_width:
                    # Full-width: wide & tall enough to breathe
                    fig_w = 10
                    fig_h = max(4.0, min(6.0, 3.0 + n_bars * 0.12))
                else:
                    # Multi-column: compact but scale height with bar count
                    fig_w = 5.0
                    fig_h = max(3.2, min(5.5, 2.8 + n_bars * 0.10))

                # Font scale: smaller when many bars, larger when few
                label_fs  = max(6, min(9,  10 - n_bars // 6))
                tick_fs   = max(6, min(8,   9 - n_bars // 8))
                val_fs    = max(5, min(8,   9 - n_bars // 7))
                # ─────────────────────────────────────────────────────────────

                ctx = chart_containers[0] if use_full_width else chart_containers[idx % 3]
                with ctx:
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    
                    if is_aggregated:
                        bar_data = self._sort_bar_data_by_category(bar_data, cat_col, col)
                        
                        bars = ax.bar(range(len(bar_data)), bar_data[col].values,
                                     color=CHART_COLORS[0], alpha=0.95, edgecolor=BG_PANEL, linewidth=1.2)
                        
                        ax.set_xticks(range(len(bar_data)))
                        rotation = 45 if n_bars > 6 else 0
                        ax.set_xticklabels(bar_data[cat_col].values, rotation=rotation,
                                           ha='right' if rotation else 'center', fontsize=tick_fs)
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', fontsize=val_fs,
                                    fontweight='bold', color=TEXT_MAIN)
                        
                        ax.set_title(f"{col} by {cat_col}", fontsize=11 if use_full_width else 10,
                                     fontweight="bold")
                        ax.set_xlabel(cat_col, fontsize=label_fs, fontweight='bold')
                        ax.set_ylabel(col, fontsize=label_fs, fontweight='bold')
                    else:
                        counts, bins = np.histogram(col_data, bins=15)
                        
                        bars = ax.bar(range(len(counts)), counts, color=CHART_COLORS[0],
                                     alpha=0.99, edgecolor=BG_PANEL, linewidth=1.2)
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', fontsize=val_fs,
                                    fontweight='bold', color=TEXT_MAIN)
                        
                        ax.set_title(f"{col} Distribution", fontsize=11 if use_full_width else 10,
                                     fontweight="bold")
                        ax.set_xlabel(f"{col} Value Ranges", fontsize=label_fs, fontweight='bold')
                        ax.set_ylabel("Frequency (Count)", fontsize=label_fs, fontweight='bold')
                    
                    ax.grid(axis="y", alpha=0.3, linestyle="--")
                    ax.set_axisbelow(True)
                    plt.tight_layout()
                    show_centered_plot(fig, use_container_width=True, clear_figure=True)
    
    def render_correlation_analysis(self):
        """Render correlation analysis for numeric columns"""
        if len(self.insights.numeric_cols) < 2:
            return
        
        corr_matrix = self.insights.get_correlations()
        
        if corr_matrix is None or corr_matrix.empty:
            return
        
        st.subheader(" Correlation Analysis")
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for "strong" correlation
                    strong_corr.append({
                        "Variable 1": corr_matrix.columns[i],
                        "Variable 2": corr_matrix.columns[j],
                        "Correlation": round(corr_val, 3)
                    })
        
        # Display correlations
        if strong_corr:
            with st.expander(" Strong Correlations (|r| > 0.5)", expanded=True):
                corr_df = pd.DataFrame(strong_corr).sort_values("Correlation", key=abs, ascending=True)
                st.dataframe(corr_df, use_container_width=True)
        
        # Heatmap
        with st.expander("Correlation Heatmap", expanded=False):
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                       cbar_kws={"label": "Correlation"}, ax=ax, square=True,
                       annot_kws={"size": 8})
            ax.set_title("Numeric Variables Correlation Matrix", fontsize=10, fontweight="bold")
            plt.tight_layout()
            show_centered_plot(fig, use_container_width=True)
    
    def render_trend_analysis(self):
        """Render trend analysis if date and numeric columns exist"""
        trend_info = self.insights.detect_trends()
        
        if not trend_info:
            return
        
        date_col, numeric_col = trend_info
        
        st.subheader("Trend Analysis")
        
        try:
            trend_df = self.df[[date_col, numeric_col]].copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
            trend_df = trend_df.dropna(subset=[date_col, numeric_col])
            trend_df = trend_df.sort_values(date_col)
            
            if len(trend_df) >= 2:
                fig, ax = plt.subplots(figsize=(7.5, 3.2))
                ax.plot(trend_df[date_col], trend_df[numeric_col], marker='o', linewidth=2.2,
                       color=PRIMARY, markersize=5, markerfacecolor=ACCENT2, markeredgecolor=PRIMARY)
                ax.set_xlabel(date_col, fontsize=8)
                ax.set_ylabel(numeric_col, fontsize=8)
                ax.set_title(f"{numeric_col} Over Time ({date_col})", fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                show_centered_plot(fig, use_container_width=True)
        except Exception as e:
            logger.warning(f"Could not render trend analysis: {e}")
    
    def render_categorical_analysis(self):
        """Render categorical column analysis with pie charts only"""
        cat_insights = self.insights.get_categorical_insights()
        
        if not cat_insights:
            return
        
        # st.subheader(" Categorical Columns Analysis")
        
        # Focus on key categorical columns with reasonable cardinality
        key_categorical = [col for col in self.insights.categorical_cols 
                          if cat_insights[col]["unique_count"] <= 20][:3]
        
        # Fallback if no low-cardinality categorical columns exist, but there are categorical columns
        if not key_categorical and self.insights.categorical_cols:
            key_categorical = self.insights.categorical_cols[:1]
        
        if key_categorical:
            st.subheader(" Category Distributions (Pie Charts)")

            num_pies = len(key_categorical)
            use_full_width = (num_pies == 1)
            if use_full_width:
                pie_containers = [st.container()]
            else:
                pie_containers = st.columns(min(3, num_pies))
            
            for idx, col in enumerate(key_categorical):
                # Resolve value column
                value_col = None
                if len(self.insights.numeric_cols) == 1 and len(self.insights.categorical_cols) <= 2:
                    value_col = self.insights.numeric_cols[0]
                if value_col is None:
                    possible_value_cols = [c for c in self.insights.numeric_cols 
                                         if any(x in c.lower() for x in ['count', 'total', 'frequency', 'value', 'sum'])]
                    if possible_value_cols:
                        value_col = possible_value_cols[0]
                
                if value_col:
                    pie_data = self._aggregate_by_category(col, value_col).sort_values(ascending=False)
                else:
                    pie_data = self.df[col].astype(str).value_counts()
                
                # Group high-cardinality items (if > 10 categories) into "Other"
                if len(pie_data) > 10:
                    top_n = pie_data.head(9)
                    other_sum = pie_data.iloc[9:].sum()
                    pie_data = pd.concat([top_n, pd.Series({"Other": other_sum})])
                
                if len(pie_data) == 0:
                    continue

                # ── Adaptive figsize & font scale by slice count ─────────────
                n_slices = len(pie_data)
                if use_full_width:
                    fig_w = fig_h = 0.5 * max(6.0, min(9.0, 5.0 + n_slices * 0.2))
                else:
                    fig_w = fig_h = 0.5 * max(4.2, min(6.5, 3.8 + n_slices * 0.18))

                label_fs  = max(5, min(8, 9 - n_slices // 3))
                pct_fs    = max(5, min(7, 8 - n_slices // 4))
                title_fs  = 10 if use_full_width else 8
                # ─────────────────────────────────────────────────────────────

                ctx = pie_containers[0] if use_full_width else pie_containers[idx % 3]
                with ctx:
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    colors = CHART_COLORS[:n_slices]

                    # For many slices suppress labels on tiny wedges to avoid overlap
                    display_labels = pie_data.index.tolist()
                    if n_slices > 8:
                        threshold = pie_data.sum() * 0.03
                        display_labels = [
                            lbl if val >= threshold else ''
                            for lbl, val in zip(pie_data.index, pie_data.values)
                        ]

                    total = pie_data.sum()

                    # Custom autopct: show count on line 1, pct on line 2
                    def make_autopct(values):
                        def autopct(pct):
                            count = int(round(pct / 100.0 * total))
                            return f"{count:,}\n({pct:.1f}%)"
                        return autopct

                    wedges, texts, autotexts = ax.pie(
                        pie_data.values,
                        labels=display_labels,
                        autopct=make_autopct(pie_data.values),
                        startangle=90,
                        colors=colors,
                        textprops={'fontsize': label_fs},
                        pctdistance=0.78,
                    )
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(pct_fs)

                    title = (
                        f"{col} Distribution (by {value_col})"
                        if value_col else f"{col} Distribution"
                    )
                    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=12)
                    ax.axis("equal")

                    # Legend with count + pct for every slice
                    legend_labels = [
                        f"{lbl}:  {int(val):,}  ({val / total * 100:.1f}%)"
                        for lbl, val in zip(pie_data.index, pie_data.values)
                    ]
                    ax.legend(
                        wedges, legend_labels,
                        title="Category", title_fontsize=max(5, pct_fs - 1),
                        loc="lower center",
                            bbox_to_anchor=(0.5, -0.25),
                        ncol=min(3, n_slices),
                        fontsize=max(4, pct_fs - 2),
                        frameon=True,
                        framealpha=0.85,
                        edgecolor=GRID_LINE,
                    )
                    plt.tight_layout()
                    if use_full_width:
                        show_centered_plot(fig, use_container_width=True, clear_figure=True, ratio=0.45)
                    else:
                        st.pyplot(fig, use_container_width=False, clear_figure=True)
    
    def render_line_chart(self):
        """Render a line / area chart for ordered/time-series categorical data"""
        if not self.insights.numeric_cols:
            return
        num_col = self.insights.numeric_cols[0]
        
        # Support multi-series line chart if multiple categorical columns are present
        if len(self.insights.categorical_cols) >= 2:
            c1 = self.insights.categorical_cols[0]
            c2 = self.insights.categorical_cols[1]
            
            # Sort by time-series column c1 and aggregate values
            num_lower = str(num_col).lower()
            agg_func = 'mean' if any(x in num_lower for x in ["avg", "mean", "score", "pct", "ratio", "percentage"]) else 'sum'
            plot_df = self.df.groupby([c1, c2])[num_col].agg(agg_func).reset_index()

            if self._is_year_column(c1, plot_df[c1]):
                plot_df["_sort_key"] = pd.to_numeric(plot_df[c1], errors="coerce")
                plot_df = plot_df.sort_values("_sort_key").drop(columns=["_sort_key"])
            else:
                plot_df = plot_df.sort_values(c1)
                
            n = plot_df[c1].nunique()
            fig_w = max(9, min(16, 7 + n * 0.35))
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            
            # Draw line plot
            sns.lineplot(data=plot_df, x=c1, y=num_col, hue=c2, ax=ax, marker='o', linewidth=2.5, palette=CHART_COLORS)
            
            ax.set_title(f"{num_col} by {c1} and {c2}", fontsize=11, fontweight='bold', pad=14)
            ax.set_ylabel(num_col, fontsize=9, fontweight='bold')
            ax.set_xlabel(c1, fontsize=9, fontweight='bold')
            ax.grid(axis='y', alpha=0.25, linestyle='--')
            ax.legend(title=c2, fontsize=8, title_fontsize=9)
            plt.xticks(rotation=45 if n > 8 else 0)
            plt.tight_layout()
            show_centered_plot(fig, use_container_width=True, clear_figure=True)
            return

        cat_col = self.insights.categorical_cols[0] if self.insights.categorical_cols else None

        if cat_col:
            agg_series = self._aggregate_by_category(cat_col, num_col)
            plot_df = pd.DataFrame({cat_col: agg_series.index, num_col: agg_series.values})
            plot_df = self._sort_bar_data_by_category(plot_df, cat_col, num_col)
            x_vals = plot_df[cat_col].astype(str).values
            y_vals = plot_df[num_col].values.astype(float)
        else:
            x_vals = self.df.index.astype(str).values
            y_vals = self.df[num_col].values.astype(float)

        n = len(x_vals)

        # Adaptive figsize & font scale by series count
        if n > 10:
            fig_w = max(9, min(16, 7 + n * 0.35))
            fig_h = 5.5
        else:
            fig_w = 7.5
            fig_h = 5.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Area fill + line + markers
        ax.fill_between(range(n), y_vals, alpha=0.13, color=PRIMARY)
        ax.plot(range(n), y_vals, marker='o', linewidth=2.5,
                color=PRIMARY, markersize=7,
                markerfacecolor=ACCENT2, markeredgecolor=PRIMARY,
                markeredgewidth=1.5, zorder=3)

        label_fs = max(7, min(9, 11 - n // 5))
        y_range = float(y_vals.max() - y_vals.min()) if y_vals.max() != y_vals.min() else 1.0
        base_offset = max(12, int(30 - n * 0.5))

        for i, (xi, yi) in enumerate(zip(range(n), y_vals)):
            above = (i % 2 == 0) if n > 10 else True
            vert_offset = base_offset if above else -base_offset - 8
            va = 'bottom' if above else 'top'

            ax.annotate(
                f"{int(yi):,}",
                xy=(xi, yi),
                xytext=(0, vert_offset),
                textcoords='offset points',
                ha='center',
                va=va,
                fontsize=label_fs,
                fontweight='bold',
                color=TEXT_MAIN,
                zorder=5,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='white',
                    edgecolor=GRID_LINE,
                    linewidth=0.7,
                    alpha=0.88,
                ),
            )

        y_min = float(y_vals.min())
        y_max = float(y_vals.max())
        padding = y_range * 0.22 if n <= 10 else y_range * 0.28
        ax.set_ylim(max(0, y_min - y_range * 0.08), y_max + padding)

        ax.set_xticks(range(n))
        rotation = 45 if n > 8 else 0
        ax.set_xticklabels(
            x_vals, rotation=rotation,
            ha='right' if rotation else 'center', fontsize=max(7, 9 - n // 8)
        )
        ax.set_title(
            f"{num_col} — Line / Area Chart" + (f" by {cat_col}" if cat_col else ""),
            fontsize=12, fontweight='bold', pad=14
        )
        ax.set_ylabel(num_col, fontsize=9)
        ax.set_xlabel(cat_col if cat_col else "", fontsize=9)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=1.5)
        show_centered_plot(fig, use_container_width=True, clear_figure=True)

    def render_grouped_bar(self):
        """Render a grouped bar chart for 2 categorical columns and 1 numeric column"""
        if len(self.insights.categorical_cols) < 2 or not self.insights.numeric_cols:
            self.render_numeric_analysis()
            return
        
        c1 = self.insights.categorical_cols[0]
        c2 = self.insights.categorical_cols[1]
        val = self.insights.numeric_cols[0]
        
        # Limit cardinality of c1 to top 15 to keep the chart clean
        top_c1 = self.df[c1].value_counts().head(15).index
        plot_df = self.df[self.df[c1].isin(top_c1)].copy()
        
        n_groups = plot_df[c1].nunique()
        fig_w = max(8, min(16, 6 + n_groups * 0.5))
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        
        sns.barplot(data=plot_df, x=c1, y=val, hue=c2, ax=ax, palette=CHART_COLORS)
        
        ax.set_title(f"{val} distribution of {c2} for each {c1}", fontsize=11, fontweight='bold')
        ax.set_xlabel(c1, fontsize=9, fontweight='bold')
        ax.set_ylabel(val, fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(title=c2, fontsize=8, title_fontsize=9)
        plt.xticks(rotation=45 if n_groups > 6 else 0)
        plt.tight_layout()
        show_centered_plot(fig, use_container_width=True, clear_figure=True)

    def render_horizontal_bar(self):
        """Render a horizontal bar chart — best when category labels are long"""
        if not self.insights.numeric_cols or not self.insights.categorical_cols:
            self.render_numeric_analysis()
            return
        num_col = self.insights.numeric_cols[0]
        cat_col = self.insights.categorical_cols[0]
        agg_series = self._aggregate_by_category(cat_col, num_col).sort_values(ascending=True)
        bar_df = pd.DataFrame({cat_col: agg_series.index, num_col: agg_series.values})
        n = len(bar_df)
        fig_h = max(4, min(12, 2.5 + n * 0.32))
        fig, ax = plt.subplots(figsize=(9, fig_h))
        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(n)]
        bars = ax.barh(range(n), bar_df[num_col].values, color=colors, edgecolor=BG_PANEL, height=0.65)
        ax.set_yticks(range(n))
        ax.set_yticklabels(bar_df[cat_col].astype(str).values, fontsize=max(6, 9 - n // 8))
        for bar in bars:
            w = bar.get_width()
            ax.text(w, bar.get_y() + bar.get_height() / 2.,
                    f" {int(w):,}", va='center', fontsize=7, fontweight='bold', color=TEXT_MAIN)
        ax.set_xlabel(num_col, fontsize=9)
        ax.set_title(f"{num_col} by {cat_col} — Horizontal Bar", fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        show_centered_plot(fig, use_container_width=True, clear_figure=True)

    def render_scatter(self):
        """Render a scatter plot between the first two numeric columns"""
        if len(self.insights.numeric_cols) < 2:
            st.info("Scatter plot requires at least 2 numeric columns. Showing bar chart instead.")
            self.render_numeric_analysis()
            return
        x_col, y_col = self.insights.numeric_cols[0], self.insights.numeric_cols[1]
        color_col = self.insights.categorical_cols[0] if self.insights.categorical_cols else None
        fig, ax = plt.subplots(figsize=(8, 5))
        if color_col:
            cats = self.df[color_col].astype(str).unique()
            for i, cat in enumerate(cats):
                mask = self.df[color_col].astype(str) == cat
                ax.scatter(self.df.loc[mask, x_col], self.df.loc[mask, y_col],
                           label=cat, color=CHART_COLORS[i % len(CHART_COLORS)], alpha=0.75, s=60)
            ax.legend(fontsize=8, title=color_col, framealpha=0.8)
        else:
            ax.scatter(self.df[x_col], self.df[y_col], color=PRIMARY, alpha=0.75, s=60)
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_title(f"{x_col} vs {y_col} — Scatter Plot", fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        show_centered_plot(fig, use_container_width=True, clear_figure=True)

    def render_donut(self):
        """Render a donut chart (pie with hole) for categorical data"""
        if not self.insights.categorical_cols:
            st.info("Donut chart needs categorical data. Showing bar chart instead.")
            self.render_numeric_analysis()
            return
        cat_col = self.insights.categorical_cols[0]
        val_col = self.insights.numeric_cols[0] if self.insights.numeric_cols else None
        if val_col:
            pie_data = self._aggregate_by_category(cat_col, val_col).sort_values(ascending=False)
        else:
            pie_data = self.df[cat_col].astype(str).value_counts()
            
        # Group high-cardinality items (if > 10 categories) into "Other"
        if len(pie_data) > 10:
            top_n = pie_data.head(9)
            other_sum = pie_data.iloc[9:].sum()
            pie_data = pd.concat([top_n, pd.Series({"Other": other_sum})])
            
        n = len(pie_data)
        fig_sz = 0.5 * max(5.5, min(8.5, 4.5 + n * 0.2))
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
        total = pie_data.sum()
        colors = CHART_COLORS[:n]
        wedges, texts, autotexts = ax.pie(
            pie_data.values,
            labels=pie_data.index.astype(str),
            autopct=lambda pct: f"{int(round(pct/100*total)):,}\n({pct:.1f}%)",
            startangle=90, colors=colors,
            wedgeprops=dict(width=0.55),
            pctdistance=0.75,
            textprops={'fontsize': max(5, 8 - n // 3)}
        )
        for at in autotexts:
            at.set_fontsize(max(5, 7 - n // 4))
            at.set_fontweight('bold')
            at.set_color('white')
        ax.set_title(f"{cat_col} — Donut Chart", fontsize=10, fontweight='bold', pad=14)
        ax.axis('equal')
        plt.tight_layout()
        show_centered_plot(fig, use_container_width=True, clear_figure=True, ratio=0.4)

    def render_histogram(self):
        """Render histogram(s) for numeric column distributions"""
        if not self.insights.numeric_cols:
            st.info("No numeric columns found for histogram.")
            return
        cols_to_plot = self.insights.numeric_cols[:3]
        n_plots = len(cols_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.2))
        if n_plots == 1:
            axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            data = self.df[col].dropna()
            n_bins = min(30, max(10, len(data) // 5))
            ax.hist(data, bins=n_bins, color=PRIMARY, edgecolor=BG_PANEL, alpha=0.9, linewidth=0.8)
            mean_v = data.mean()
            ax.axvline(mean_v, color=ACCENT2, linestyle='--', linewidth=1.8, label=f'Mean: {mean_v:.1f}')
            ax.legend(fontsize=8)
            ax.set_title(f"{col} — Histogram", fontsize=10, fontweight='bold')
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        show_centered_plot(fig, use_container_width=True, clear_figure=True)

    def render_heatmap(self):
        """Render correlation heatmap (needs 2+ numeric columns) or a pivot heatmap"""
        if len(self.insights.numeric_cols) >= 2:
            self.render_correlation_analysis()
        elif len(self.insights.categorical_cols) >= 2 and self.insights.numeric_cols:
            # Pivot heatmap: cat1 × cat2 → numeric
            c1, c2 = self.insights.categorical_cols[0], self.insights.categorical_cols[1]
            val = self.insights.numeric_cols[0]
            try:
                pivot = self.df.pivot_table(index=c1, columns=c2, values=val, aggfunc='sum', fill_value=0)
                fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.9), max(4, len(pivot) * 0.7)))
                sns.heatmap(pivot, annot=True, fmt='g', cmap='Blues', ax=ax,
                            linewidths=0.5, linecolor=GRID_LINE, annot_kws={'size': 8})
                ax.set_title(f"{val} by {c1} × {c2} — Heatmap", fontsize=11, fontweight='bold')
                plt.tight_layout()
                show_centered_plot(fig, use_container_width=True, clear_figure=True)
            except Exception as e:
                logger.warning(f"Pivot heatmap failed: {e}")
                self.render_numeric_analysis()
        else:
            st.info("Heatmap needs 2+ numeric or 2 categorical columns. Showing bar chart instead.")
            self.render_numeric_analysis()

    def _decide_chart_type_with_llm(self, user_query: str) -> str:
        """Use LLM to dynamically decide the best chart type based on the user's question."""
        prompt = f"""You are an expert data visualization assistant.
Based on the user's query and the data properties, decide the single most appropriate chart type to display.

USER QUERY: "{user_query}"

AVAILABLE DATA:
- Numeric columns: {', '.join(self.insights.numeric_cols) if self.insights.numeric_cols else 'None'}
- Categorical columns: {', '.join(self.insights.categorical_cols) if self.insights.categorical_cols else 'None'}
- Date columns: {', '.join(self.insights.date_cols) if self.insights.date_cols else 'None'}

CHART TYPE OPTIONS:
1. "bar" - Best for comparing numeric values across categories, or showing distributions of numeric data.
2. "pie" - Best for showing proportions or percentages of categorical data (only if categorical columns exist).
3. "trend" - Best for showing changes over time (only if both date and numeric columns exist).
4. "correlation" - Best for showing relationships between multiple numeric variables (only if 2+ numeric columns exist).
5. "auto" - If you are unsure or want to display all relevant charts.

Respond with ONLY ONE word from the options above: bar, pie, trend, correlation, or auto.
"""
        try:
            response = call_llm_with_cache(prompt, temperature=0.1)
            if response:
                choice = response.strip().lower()
                for valid_choice in ["bar", "pie", "trend", "correlation", "auto"]:
                    if valid_choice in choice:
                        return valid_choice
            return "auto"
        except Exception as e:
            logger.warning("LLM chart detection failed: %s", e)
            return "auto"

    def render_insights(self, chart_type: str = "auto", user_query: str = None):
        """Main method to render all insights"""
        # Executive summary always shown
        self.render_executive_summary()
        st.divider()

        # LLM-driven chart type selection when auto
        if chart_type == "auto" and user_query:
            llm_chart_type = self._decide_chart_type_with_llm(user_query)
            if llm_chart_type != "auto":
                st.info(f" Selecting '{llm_chart_type}' chart based on your query.")
                chart_type = llm_chart_type

        # ── Dispatch to the correct renderer ─────────────────────────────────
        if chart_type == "bar":
            if len(self.insights.categorical_cols) >= 2 and self.insights.numeric_cols:
                self.render_grouped_bar()
            elif self.insights.numeric_cols:
                self.render_numeric_analysis()
            else:
                st.info("No numeric columns found. Showing categorical distributions instead.")
                self.render_categorical_analysis()

        elif chart_type == "pie":
            if self.insights.categorical_cols:
                self.render_categorical_analysis()
            else:
                st.info("No categorical columns found. Showing numeric bar charts instead.")
                self.render_numeric_analysis()

        elif chart_type == "donut":
            self.render_donut()

        elif chart_type in ("line", "area"):
            self.render_line_chart()

        elif chart_type == "horizontal_bar":
            self.render_horizontal_bar()

        elif chart_type == "scatter":
            self.render_scatter()

        elif chart_type == "histogram":
            self.render_histogram()

        elif chart_type == "heatmap":
            self.render_heatmap()

        elif chart_type == "trend":
            if self.insights.date_cols and self.insights.numeric_cols:
                self.render_trend_analysis()
            else:
                st.info("Missing date or numeric columns for trend. Showing line chart instead.")
                self.render_line_chart()

        elif chart_type == "correlation":
            if len(self.insights.numeric_cols) >= 2:
                self.render_correlation_analysis()
            else:
                st.info("Need 2+ numeric columns for correlation. Showing all relevant charts.")
                chart_type = "auto"

        if chart_type == "auto":
            if len(self.insights.categorical_cols) >= 2 and self.insights.numeric_cols:
                self.render_grouped_bar()
                st.divider()
                self.render_heatmap()
                st.divider()
            else:
                if self.insights.numeric_cols:
                    self.render_numeric_analysis()
                    st.divider()
                if self.insights.categorical_cols:
                    self.render_categorical_analysis()
                    st.divider()
            if len(self.insights.numeric_cols) >= 2:
                self.render_correlation_analysis()
                st.divider()
            if self.insights.date_cols and self.insights.numeric_cols:
                self.render_trend_analysis()
                st.divider()




def generate_insights(result_df: pd.DataFrame, chart_type: str = "auto", user_query: str = None) -> None:
    """
    Main entry point for generating insights
    
    Args:
        result_df: DataFrame with query results
        chart_type: Chart type to render
        user_query: Original user query from the user to dynamically determine chart representations
    """
    if result_df is None or result_df.empty:
        return
    
    try:
        # Create insights generator and visualizer
        insights_gen = DataInsightsGenerator(result_df)
        visualizer = InsightVisualizer(result_df, insights_gen)
        
        # Render all insights
        visualizer.render_insights(chart_type=chart_type, user_query=user_query)
        
    except Exception as e:
        logger.exception(f"Error generating insights: {e}")
        st.error(f"Could not generate insights: {str(e)}")
