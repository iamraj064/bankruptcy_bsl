import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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


class DataInsightsGenerator:
    """Generate comprehensive insights from query results"""
    
    def __init__(self, result_df: pd.DataFrame):
        self.df = result_df
        self._chart_counter = 0  # counter for unique plotly_chart keys
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

    def _chart_idx(self) -> int:
        """Return a unique incrementing index for each plotly_chart key."""
        self._chart_counter += 1
        return self._chart_counter

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


# Ã¢â€â‚¬Ã¢â€â‚¬ Shared Plotly Layout Helper Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
def _plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 420,
) -> dict:
    """Return a consistent, professional Plotly layout dictionary."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=16, color=TEXT_MAIN, family="Outfit, Inter, sans-serif"),
            x=0.02, xanchor='left',
        ),
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=13, color=TEXT_MAIN)),
            tickfont=dict(size=11, color=TEXT_MUTED),
            showgrid=True, gridcolor=GRID_LINE, gridwidth=1,
            linecolor=GRID_LINE,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=13, color=TEXT_MAIN)),
            tickfont=dict(size=11, color=TEXT_MUTED),
            showgrid=True, gridcolor=GRID_LINE, gridwidth=1,
            linecolor=GRID_LINE,
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        font=dict(family="Inter, sans-serif", color=TEXT_MAIN),
        hoverlabel=dict(
            bgcolor='white', bordercolor=GRID_LINE,
            font=dict(size=13, color=TEXT_MAIN),
        ),
    )


class InsightVisualizer:
    """Handle visualization of insights with smart recommendations"""
    
    def __init__(self, result_df: pd.DataFrame, insights_gen: DataInsightsGenerator):
        self.df = result_df
        self.insights = insights_gen
        self._chart_counter = 0
        import uuid
        self.viz_id = uuid.uuid4().hex[:8]

    def _chart_idx(self) -> int:
        """Return a unique incrementing index for each plotly_chart key."""
        self._chart_counter += 1
        return self._chart_counter

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
        user_query: str = None,
    ) -> str:
        """Build a prompt for the LLM summarizing dataset trends and drivers."""
        prompt_lines = [
            "Role: You are an elite strategic credit risk advisor and business intelligence analyst.",
            "Task: Generate a highly polished, client-focused 'Trend Summary Analysis' based on the dataset characteristics provided below.",
            "",
            "CRITICAL FORMATTING RULES (follow exactly):",
            "1. Tone: Professional, executive-level, clear, and highly authoritative. Speak directly to business leaders/clients.",
            "2. Use the exact markdown structure shown below Ã¢â‚¬â€ do NOT merge sections into a single paragraph.",
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

        if user_query:
            prompt_lines.insert(2, f"ORIGINAL USER QUESTION/QUERY: \"{user_query}\"")
            prompt_lines.insert(3, "INSTRUCTIONS FOR THIS QUERY:")
            prompt_lines.insert(4, "1. Answer the user question directly and accurately using the numbers in the dataset.")
            prompt_lines.insert(5, "2. If the user's question involves a comparison (e.g. 'compare', 'vs', 'difference', 'increase/decrease', or comparing chapters/states/times):")
            prompt_lines.insert(6, "   - You MUST perform a comparison: calculate the absolute and/or percentage change/difference between the key categories/KPIs.")
            prompt_lines.insert(7, "   - Identify and list the main driving factors behind these differences or trends.")
            prompt_lines.insert(8, "3. Ensure all three sections are highly accurate, sharp, specific, and directly informed by the actual data rows provided below. Avoid generic boilerplate phrasing.")
            prompt_lines.insert(9, "")

        # Format up to 50 rows of data as CSV for precise LLM grounding
        df_sample = self.df.head(50).to_csv(index=False)
        prompt_lines.append("")
        prompt_lines.append("ACTUAL DATASET SAMPLE (Extract exact values and compare these directly):")
        prompt_lines.append("```csv")
        prompt_lines.append(df_sample)
        prompt_lines.append("```")
        prompt_lines.append("")

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

    def _refine_summary_for_enduser(self, raw_summary: str, user_query: str = None) -> str:
        """Post-process/refine the raw summary text to ensure premium, end-user focused sentence structure."""
        user_query_info = f"USER QUERY: \"{user_query}\"\n\n" if user_query else ""
        prompt = (
            "You are a Senior Executive Editor specializing in corporate reports and business intelligence.\n"
            "Your task is to take the draft analysis below and rewrite/polish it to make it sound highly professional, "
            "compelling, and polished for executive end-users. \n\n"
            f"{user_query_info}"
            "DRAFT ANALYSIS:\n"
            f"{raw_summary}\n\n"
            "REWRITING & POLISHING RULES:\n"
            "1. Strictly maintain the exact markdown headers (e.g. '## Executive Trend Overview', '## Key Drivers & Concentrations', '## Strategic Client Implications') and bullet point list structure.\n"
            "2. Improve sentence formation: make sentences flow naturally, sound elegant, premium, and authoritative. Avoid clunky, repetitive, or robotic sentence structures.\n"
            "3. Ensure the tone is highly tailored for business leaders, decision-makers, and clients. Use executive-level vocabulary (e.g. concentration risk, portfolio optimization, credit risk exposure) instead of generic phrases.\n"
            "4. Retain all key statistics, numbers, percentages, and names from the draft. Do not invent or change any figures. Ensure comparative metrics like percentage differences are preserved.\n"
            "5. Keep the total length concise, executive-focused, and under 160 words.\n"
            "6. Output ONLY the polished markdown. Do not include any intro, outro, meta-commentary, or surrounding markdown code blocks (e.g. do not wrap in ```markdown or ```)."
        )
        try:
            refined_response = call_llm_with_cache(prompt, temperature=0.6)
            return refined_response.strip() if refined_response else raw_summary
        except Exception as e:
            logger.warning("Refining summary for end-user failed: %s", e)
            return raw_summary

    def _generate_llm_summary(self, stats: Dict, user_query: str = None) -> Optional[str]:
        """Generate an LLM-based executive summary of dataset insights."""
        try:
            numeric_insights = self.insights.get_numeric_insights()
            categorical_insights = self.insights.get_categorical_insights()
            prompt = self._build_llm_summary_prompt(stats, numeric_insights, categorical_insights, user_query=user_query)
            llm_response = call_llm_with_cache(prompt, temperature=0.1)
            if llm_response:
                return self._refine_summary_for_enduser(llm_response.strip(), user_query=user_query)
            return None
        except Exception as e:
            logger.warning("LLM summary generation failed: %s", e)
            return None

    def render_executive_summary(self, user_query: str = None):
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

        llm_summary = self._generate_llm_summary(stats, user_query=user_query)
        if llm_summary:
            import re
            # Scale heading font sizes to 60% of Streamlit defaults:
            # ## (h2) default ~1.75rem Ã¢â€ â€™ 1.05rem | ### (h3) default ~1.25rem Ã¢â€ â€™ 0.75rem
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
        """Render comprehensive numeric column analysis using Plotly"""
        numeric_insights = self.insights.get_numeric_insights()
        if not numeric_insights:
            return

        key_numeric = self.insights.numeric_cols[:3]
        if not key_numeric:
            return

        for col in key_numeric:
            col_data = self.df[col].dropna()
            if len(col_data) < 2:
                continue

            is_aggregated = (
                len(self.insights.categorical_cols) >= 1
                and len(self.insights.numeric_cols) == 1
            )

            if is_aggregated:
                cat_col = self.insights.categorical_cols[0]
                agg_series = self._aggregate_by_category(cat_col, col)
                bar_data = pd.DataFrame({cat_col: agg_series.index, col: agg_series.values})
                bar_data = self._sort_bar_data_by_category(bar_data, cat_col, col)

                # Clean column name for display
                x_label = cat_col.replace('_', ' ').title()
                y_label = col.replace('_', ' ').title()

                fig = px.bar(
                    bar_data, x=cat_col, y=col,
                    text=bar_data[col].apply(lambda v: f"{int(v):,}"),
                    color_discrete_sequence=[CHART_COLORS[0]],
                    labels={cat_col: x_label, col: y_label},
                )
                fig.update_traces(
                    textposition='outside',
                    marker_line_width=0,
                    hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y:,}}<extra></extra>"
                )
                fig.update_layout(
                    **_plotly_layout(
                        title=f"{y_label} by {x_label}",
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                    )
                )
                fig.update_xaxes(type='category', tickangle=-35 if bar_data[cat_col].nunique() > 6 else 0)
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
            else:
                # Histogram
                fig = px.histogram(
                    self.df, x=col,
                    nbins=min(30, max(10, len(col_data) // 5)),
                    color_discrete_sequence=[CHART_COLORS[0]],
                    labels={col: col.replace('_', ' ').title()},
                )
                mean_v = float(col_data.mean())
                fig.add_vline(
                    x=mean_v, line_dash="dash", line_color=CHART_COLORS[1], line_width=2,
                    annotation_text=f"Mean: {mean_v:,.1f}",
                    annotation_position="top right"
                )
                fig.update_layout(
                    **_plotly_layout(
                        title=f"{col.replace('_', ' ').title()} Distribution",
                        xaxis_title=col.replace('_', ' ').title(),
                        yaxis_title="Frequency",
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
    
    def render_correlation_analysis(self):
        """Render correlation analysis using Plotly heatmap"""
        if len(self.insights.numeric_cols) < 2:
            return

        corr_matrix = self.insights.get_correlations()
        if corr_matrix is None or corr_matrix.empty:
            return

        st.subheader(" Correlation Analysis")

        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        "Variable 1": corr_matrix.columns[i],
                        "Variable 2": corr_matrix.columns[j],
                        "Correlation": round(corr_val, 3)
                    })

        if strong_corr:
            with st.expander(" Strong Correlations (|r| > 0.5)", expanded=True):
                corr_df = pd.DataFrame(strong_corr).sort_values("Correlation", key=abs, ascending=False)
                st.dataframe(corr_df, use_container_width=True)

        with st.expander("Correlation Heatmap", expanded=True):
            labels = [c.replace('_', ' ').title() for c in corr_matrix.columns]
            fig = px.imshow(
                corr_matrix.values,
                x=labels, y=labels,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto',
            )
            fig.update_layout(
                **_plotly_layout(
                    title="Numeric Variables Correlation Matrix",
                    height=max(380, len(corr_matrix) * 50),
                )
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
    
    def render_trend_analysis(self):
        """Render trend analysis using Plotly line chart"""
        trend_info = self.insights.detect_trends()
        if not trend_info:
            return
        date_col, numeric_col = trend_info
        st.subheader("Trend Analysis")
        try:
            trend_df = self.df[[date_col, numeric_col]].copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
            trend_df = trend_df.dropna(subset=[date_col, numeric_col]).sort_values(date_col)

            if len(trend_df) >= 2:
                date_label = date_col.replace('_', ' ').title()
                num_label = numeric_col.replace('_', ' ').title()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_df[date_col], y=trend_df[numeric_col],
                    mode='lines+markers',
                    fill='tozeroy',
                    fillcolor='rgba(37,99,235,0.08)',
                    line=dict(color=CHART_COLORS[0], width=2.5),
                    marker=dict(size=7, color=CHART_COLORS[1]),
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{num_label}: %{{y:,}}<extra></extra>"
                ))
                fig.update_layout(
                    **_plotly_layout(
                        title=f"{num_label} Over Time",
                        xaxis_title=date_label,
                        yaxis_title=num_label,
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
        except Exception as e:
            logger.warning(f"Could not render trend analysis: {e}")

    def render_categorical_analysis(self):
        """Render categorical column analysis with professional Plotly pie charts"""
        cat_insights = self.insights.get_categorical_insights()
        if not cat_insights:
            return

        key_categorical = [col for col in self.insights.categorical_cols
                          if cat_insights[col]["unique_count"] <= 20][:3]
        if not key_categorical and self.insights.categorical_cols:
            key_categorical = self.insights.categorical_cols[:1]

        if not key_categorical:
            return

        st.subheader(" Category Distributions")

        num_pies = len(key_categorical)
        if num_pies == 1:
            pie_containers = [st.container()]
        else:
            pie_containers = st.columns(min(3, num_pies))

        for idx, col in enumerate(key_categorical):
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

            if len(pie_data) > 10:
                top_n = pie_data.head(9)
                other_sum = pie_data.iloc[9:].sum()
                pie_data = pd.concat([top_n, pd.Series({"Other": other_sum})])

            if len(pie_data) == 0:
                continue

            total = pie_data.sum()
            clean_col = col.replace('_', ' ').title()
            title = f"{clean_col} Distribution" + (f" (by {value_col.replace('_', ' ').title()})" if value_col else "")

            customdata = [
                f"{int(v):,} ({v / total * 100:.1f}%)" for v in pie_data.values
            ]

            fig = go.Figure(data=[go.Pie(
                labels=pie_data.index.astype(str).tolist(),
                values=pie_data.values.tolist(),
                hole=0.35,
                marker=dict(
                    colors=CHART_COLORS[:len(pie_data)],
                    line=dict(color='white', width=2)
                ),
                textinfo='percent+label',
                textposition='auto',
                textfont=dict(size=13, color='white'),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
                insidetextorientation='auto',
                pull=[0.03] * len(pie_data),
                sort=False,
            )])

            fig.update_layout(**_plotly_layout(title=title, height=400))
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation='v',
                    x=1.02, y=0.5,
                    xanchor='left',
                    font=dict(size=12, color=TEXT_MAIN),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor=GRID_LINE,
                    borderwidth=1,
                )
            )

            ctx = pie_containers[0] if num_pies == 1 else pie_containers[idx % 3]
            with ctx:
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
    
    def render_line_chart(self):
        """Render a professional Plotly line/area chart"""
        if not self.insights.numeric_cols:
            return
        num_col = self.insights.numeric_cols[0]
        y_label = num_col.replace('_', ' ').title()

        if len(self.insights.categorical_cols) >= 2:
            c1 = self.insights.categorical_cols[0]
            c2 = self.insights.categorical_cols[1]
            num_lower = str(num_col).lower()
            agg_func = 'mean' if any(x in num_lower for x in ["avg", "mean", "score", "pct", "ratio", "percentage"]) else 'sum'
            plot_df = self.df.groupby([c1, c2])[num_col].agg(agg_func).reset_index()
            if self._is_year_column(c1, plot_df[c1]):
                plot_df["_sort_key"] = pd.to_numeric(plot_df[c1], errors="coerce")
                plot_df = plot_df.sort_values("_sort_key").drop(columns=["_sort_key"])
            else:
                plot_df = plot_df.sort_values(c1)

            fig = px.line(
                plot_df, x=c1, y=num_col, color=c2,
                markers=True,
                color_discrete_sequence=CHART_COLORS,
                labels={c1: c1.replace('_', ' ').title(), num_col: y_label, c2: c2.replace('_', ' ').title()},
            )
            fig.update_traces(line_width=2.5, marker_size=7)
            fig.update_layout(
                **_plotly_layout(
                    title=f"{y_label} by {c1.replace('_', ' ').title()} & {c2.replace('_', ' ').title()}",
                    xaxis_title=c1.replace('_', ' ').title(),
                    yaxis_title=y_label,
                )
            )
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
            return

        cat_col = self.insights.categorical_cols[0] if self.insights.categorical_cols else None
        if cat_col:
            agg_series = self._aggregate_by_category(cat_col, num_col)
            plot_df = pd.DataFrame({cat_col: agg_series.index, num_col: agg_series.values})
            plot_df = self._sort_bar_data_by_category(plot_df, cat_col, num_col)
            x_vals = plot_df[cat_col].astype(str)
            y_vals = plot_df[num_col]
            x_label = cat_col.replace('_', ' ').title()
        else:
            x_vals = self.df.index.astype(str)
            y_vals = self.df[num_col]
            x_label = 'Index'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(37,99,235,0.1)',
            line=dict(color=CHART_COLORS[0], width=2.5),
            marker=dict(size=8, color=CHART_COLORS[1], line=dict(color=CHART_COLORS[0], width=1.5)),
            text=[f"{int(v):,}" for v in y_vals],
            textposition='top center',
            hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y:,}}<extra></extra>"
        ))
        fig.update_layout(
            **_plotly_layout(
                title=f"{y_label} Trend" + (f" by {cat_col.replace('_', ' ').title()}" if cat_col else ""),
                xaxis_title=x_label,
                yaxis_title=y_label,
            )
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")

    def render_grouped_bar(self):
        """Render a professional Plotly grouped bar chart"""
        if len(self.insights.categorical_cols) < 2 or not self.insights.numeric_cols:
            self.render_numeric_analysis()
            return

        c1 = self.insights.categorical_cols[0]
        c2 = self.insights.categorical_cols[1]
        val = self.insights.numeric_cols[0]

        top_c1 = self.df[c1].value_counts().head(15).index
        plot_df = self.df[self.df[c1].isin(top_c1)].copy()

        x_label = c1.replace('_', ' ').title()
        y_label = val.replace('_', ' ').title()
        hue_label = c2.replace('_', ' ').title()

        fig = px.bar(
            plot_df, x=c1, y=val, color=c2,
            barmode='group',
            color_discrete_sequence=CHART_COLORS,
            labels={c1: x_label, val: y_label, c2: hue_label},
        )
        fig.update_layout(
            **_plotly_layout(
                title=f"{y_label} by {x_label} & {hue_label}",
                xaxis_title=x_label,
                yaxis_title=y_label,
            )
        )
        fig.update_xaxes(type='category', tickangle=-35 if plot_df[c1].nunique() > 6 else 0)
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")

    def render_horizontal_bar(self):
        """Render a professional Plotly horizontal bar chart"""
        if not self.insights.numeric_cols or not self.insights.categorical_cols:
            self.render_numeric_analysis()
            return
        num_col = self.insights.numeric_cols[0]
        cat_col = self.insights.categorical_cols[0]
        agg_series = self._aggregate_by_category(cat_col, num_col).sort_values(ascending=True)
        bar_df = pd.DataFrame({cat_col: agg_series.index.astype(str), num_col: agg_series.values})

        x_label = num_col.replace('_', ' ').title()
        y_label = cat_col.replace('_', ' ').title()

        fig = px.bar(
            bar_df, x=num_col, y=cat_col,
            orientation='h',
            text=bar_df[num_col].apply(lambda v: f"{int(v):,}"),
            color=num_col,
            color_continuous_scale=[[0, CHART_COLORS[0]], [1, CHART_COLORS[1]]],
            labels={num_col: x_label, cat_col: y_label},
        )
        fig.update_traces(textposition='outside', hovertemplate=f"<b>%{{y}}</b><br>{x_label}: %{{x:,}}<extra></extra>")
        fig.update_layout(
            **_plotly_layout(
                title=f"{x_label} by {y_label}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=max(400, len(bar_df) * 32),
            ),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")

    def render_scatter(self):
        """Render a professional Plotly scatter plot"""
        if len(self.insights.numeric_cols) < 2:
            st.info("Scatter plot requires at least 2 numeric columns. Showing bar chart instead.")
            self.render_numeric_analysis()
            return
        x_col, y_col = self.insights.numeric_cols[0], self.insights.numeric_cols[1]
        color_col = self.insights.categorical_cols[0] if self.insights.categorical_cols else None

        x_label = x_col.replace('_', ' ').title()
        y_label = y_col.replace('_', ' ').title()

        fig = px.scatter(
            self.df, x=x_col, y=y_col,
            color=color_col,
            color_discrete_sequence=CHART_COLORS,
            opacity=0.75,
            labels={x_col: x_label, y_col: y_label},

        )
        fig.update_traces(marker_size=9)
        fig.update_layout(
            **_plotly_layout(
                title=f"{x_label} vs {y_label}",
                xaxis_title=x_label,
                yaxis_title=y_label,
            )
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")

    def render_donut(self):
        """Render a professional Plotly donut chart"""
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

        if len(pie_data) > 10:
            top_n = pie_data.head(9)
            other_sum = pie_data.iloc[9:].sum()
            pie_data = pd.concat([top_n, pd.Series({"Other": other_sum})])

        total = pie_data.sum()
        clean_col = cat_col.replace('_', ' ').title()

        fig = go.Figure(data=[go.Pie(
            labels=pie_data.index.astype(str).tolist(),
            values=pie_data.values.tolist(),
            hole=0.55,
            marker=dict(
                colors=CHART_COLORS[:len(pie_data)],
                line=dict(color='white', width=2)
            ),
            textinfo='percent+label',
            textposition='auto',
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            insidetextorientation='auto',
        )])
        fig.add_annotation(
            text=f"<b>{total:,}</b><br>Total",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=TEXT_MAIN),
            xanchor='center', yanchor='middle',
        )
        fig.update_layout(**_plotly_layout(title=f"{clean_col} Ã¢â‚¬â€ Donut Chart", height=420))
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation='v', x=1.02, y=0.5,
                font=dict(size=12), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID_LINE, borderwidth=1,
            )
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")

    def render_histogram(self):
        """Render professional Plotly histogram(s)"""
        if not self.insights.numeric_cols:
            st.info("No numeric columns found for histogram.")
            return
        cols_to_plot = self.insights.numeric_cols[:3]
        for col in cols_to_plot:
            data = self.df[col].dropna()
            n_bins = min(30, max(10, len(data) // 5))
            mean_v = float(data.mean())
            col_label = col.replace('_', ' ').title()

            fig = px.histogram(
                self.df, x=col, nbins=n_bins,
                color_discrete_sequence=[CHART_COLORS[0]],
                labels={col: col_label},
            )
            fig.add_vline(
                x=mean_v, line_dash="dash", line_color=CHART_COLORS[1], line_width=2,
                annotation_text=f"Mean: {mean_v:,.1f}",
                annotation_position="top right",
            )
            fig.update_layout(
                **_plotly_layout(
                    title=f"{col_label} Ã¢â‚¬â€ Histogram",
                    xaxis_title=col_label,
                    yaxis_title='Frequency',
                )
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")

    def render_heatmap(self):
        """Render professional Plotly correlation heatmap or pivot heatmap"""
        if len(self.insights.numeric_cols) >= 2:
            self.render_correlation_analysis()
        elif len(self.insights.categorical_cols) >= 2 and self.insights.numeric_cols:
            c1, c2 = self.insights.categorical_cols[0], self.insights.categorical_cols[1]
            val = self.insights.numeric_cols[0]
            try:
                pivot = self.df.pivot_table(index=c1, columns=c2, values=val, aggfunc='sum', fill_value=0)
                fig = px.imshow(
                    pivot,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    aspect='auto',
                    labels=dict(x=c2.replace('_', ' ').title(), y=c1.replace('_', ' ').title(), color=val.replace('_', ' ').title()),
                )
                fig.update_layout(
                    **_plotly_layout(
                        title=f"{val.replace('_', ' ').title()} by {c1.replace('_', ' ').title()} Ãƒâ€” {c2.replace('_', ' ').title()}",
                        height=max(400, len(pivot) * 40),
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{self.viz_id}_{self._chart_idx()}")
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

        # Ã¢â€â‚¬Ã¢â€â‚¬ Dispatch to the correct renderer Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
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


