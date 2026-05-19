# Libraries
import streamlit as st
import plotly.graph_objects as go

# Logic Import
from backend.core.insight.explain.explainer import *
from backend.core.insight.trace_log import get_log
from backend.core.session.pipeline_state import get_comparison, clear_comparison, STEP_LABELS

# UI Import

# Functions
# Design tokens - Midnight Lab Palette (Flat Modern)
BACKGROUND_COLOR       = "#24283B"  # Deep Navy Surface
BORDER_COLOR   = "#414868"          # Sharp Slate Border
TEXT_COLOR     = "#7AA2F7"          # Soft Blue for Headings
TEXT_DIM_COLOR = "#94a3b8"          # Muted Slate
ACCENT_BLUE    = "#7AA2F7"          # Soft Blue
ACCENT_INSIGHT = "#9ECE6A"          # Lime Tea
BORDER_RADIUS        = "4px"        # Minimal Sharp Corner
PADDING_STYLE      = "1.25rem"
MARGIN_GAP      = "margin: 1.5rem 0"

STEP_COLORS_INFO = {
    "Upload":              "#58a6ff",
    "Data Cleaning":       "#3fb950",
    "Data Transformation": "#d29922",
    "Model Process":          "#bc8cff",
}
PIPELINE_STEP_ORDER = ["Upload", "Data Cleaning", "Data Transformation", "Model Process"]

def render_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = (
        f'<div style="color:{TEXT_DIM_COLOR};font-size:0.95rem;margin-top:4px;font-weight:400;line-height:1.6">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="{MARGIN_GAP}">'
        f'<div style="color:{TEXT_COLOR};font-size:1.3rem;font-weight:700;letter-spacing:-0.01em">{title}</div>'
        f'{subtitle_html}</div>',
        unsafe_allow_html=True,
    )
    
def render_importance(model, X_test, y_test, task_type):
    render_section_header(
        "Feature ไหนสำคัญที่สุด?",
        "ระบบสลับค่าของแต่ละ feature แล้วดูว่า model แย่ลงแค่ไหน — แท่งยาว = สำคัญมาก",
    )

    cache_key = f"_xai_perm_{id(model)}"
    if st.session_state.get(cache_key) is None:
        with st.spinner("กำลังวิเคราะห์ความสำคัญของ feature..."):
            permutation_dataframe = compute_permutation_importance(model, X_test, y_test, task_type)
            st.session_state[cache_key] = permutation_dataframe
    permutation_dataframe = st.session_state[cache_key]

    top_features_count   = min(15, len(permutation_dataframe))
    plot_dataframe = permutation_dataframe.head(top_features_count)

    bar_chart_figure = go.Figure(go.Bar(
        x=plot_dataframe["Importance"],
        y=plot_dataframe["Feature"],
        orientation="h",
        marker=dict(color=plot_dataframe["Importance"].tolist(), colorscale=[[0, "rgba(122, 162, 247, 0.1)"], [1, "#7AA2F7"]], showscale=False),
        text=plot_dataframe["Importance"].round(3).tolist(),
        textposition="outside",
    ))
    bar_chart_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, top_features_count * 36),
        yaxis=dict(autorange="reversed"),
        xaxis_title="ความสำคัญ (ยิ่งสูง = ยิ่งสำคัญ)",
        margin=dict(t=10, b=30, l=10, r=90),
    )
    st.plotly_chart(bar_chart_figure, width="stretch")

    positive_importance_features = permutation_dataframe[permutation_dataframe["Importance"] > 0]
    if len(positive_importance_features) == 0:
        st.warning("ทุก feature มีความสำคัญน้อยมาก — ลองตรวจสอบ dataset อีกครั้ง")
        return

    render_section_header("3 Feature สำคัญที่สุด")

    rank_colors  = ["#E0AF68", "#BB9AF7", "#7AA2F7"]
    rank_labels  = ["#1", "#2", "#3"]
    total_importance = positive_importance_features["Importance"].sum() + 1e-9
    top_3_features = positive_importance_features.head(3).reset_index(drop=True)
    card_columns = st.columns(len(top_3_features))

    for index, col in enumerate(card_columns):
        row_data = top_3_features.iloc[index]
        percentage = row_data["Importance"] / total_importance * 100
        with col:
            st.markdown(
                f'<div style="background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
                f'border-top:3px solid {rank_colors[index]};border-radius:{BORDER_RADIUS};'
                f'padding:20px 16px;text-align:center">'
                f'<div style="color:{rank_colors[index]};font-size:0.85rem;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:10px">{rank_labels[index]}</div>'
                f'<div style="font-family:monospace;color:#e6edf3;font-size:1rem;'
                f'word-break:break-all;margin-bottom:12px">{row_data["Feature"]}</div>'
                f'<div style="color:{rank_colors[index]};font-size:1.6rem;font-weight:800;'
                f'line-height:1">{percentage:.0f}%</div>'
                f'<div style="color:{TEXT_DIM_COLOR};font-size:0.85rem;margin-top:6px">'
                f'ของความสำคัญรวม</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    useless_features = permutation_dataframe[permutation_dataframe["Importance"] <= 0]
    if len(useless_features):
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            f"**Feature ที่แทบไม่มีผล ({len(useless_features)} ตัว):**\n"
            + ", ".join(f"`{feature}`" for feature in useless_features["Feature"].head(5))
            + (" ..." if len(useless_features) > 5 else "")
        )

def render_trace():
    render_section_header(
        "บันทึกการตัดสินใจตลอด Pipeline",
        "เจาะลึกทุกร่องรอยการประมวลผล — ตั้งแต่ข้อมูลดิบจนถึงโมเดลที่แม่นยำที่สุด",
    )

    pipeline_log = get_log()
    if not pipeline_log:
        st.info("ยังไม่มีข้อมูล — ระบบจะเริ่มบันทึกเมื่อคุณดำเนินการผ่านแต่ละขั้นตอน")
        return

    # CSS for Timeline
    st.markdown("""<style>
.trace-container { position: relative; padding-left: 45px; margin-top: 20px; }
.trace-line { position: absolute; left: 20px; top: 0; bottom: 0; width: 2px; background: linear-gradient(180deg, #58a6ff 0%, #30363d 100%); z-index: 0; }
.step-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 30px; position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.step-dot { position: absolute; left: -35px; top: 20px; width: 24px; height: 24px; background: #0d1117; border: 3px solid #58a6ff; border-radius: 50%; z-index: 1; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #58a6ff; font-weight: 700; }
.stat-badge { background: rgba(139, 148, 158, 0.1); border: 1px solid rgba(139, 148, 158, 0.2); border-radius: 6px; padding: 6px 12px; margin-right: 8px; margin-bottom: 8px; display: inline-block; }
.insight-bubble { background: rgba(88, 166, 255, 0.05); border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)

    st.markdown('<div class="trace-container"><div class="trace-line"></div>', unsafe_allow_html=True)

    for index, entry in enumerate(pipeline_log):
        step_name = entry.get("step", "")
        log_items = entry.get("items", [])
        explanations = entry.get("explanations", [])
        color = STEP_COLORS_INFO.get(step_name, "#8b949e")

        # แยกส่วน Metrics ออกจากข้อมูลอื่นใน log_items
        metrics_html = '<div style="display: flex; flex-wrap: wrap; margin-bottom: 12px;">'
        other_items_html = ""
        
        for item in log_items:
            if ":" in item and "→" in item: # น่าจะเป็น Metric เช่น "Rows: 100 -> 80"
                label, val = item.split(":", 1)
                metrics_html += f'<div class="stat-badge"><div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase;">{label.strip()}</div><div style="font-size: 1rem; color: #e6edf3; font-weight: 600;">{val.strip()}</div></div>'
            elif not item.startswith("  "): # Header item
                other_items_html += f'<div style="font-weight: 600; color: #c9d1d9; margin: 8px 0 4px;">{item}</div>'
            else: # Detail item
                other_items_html += f'<div style="font-size: 0.85rem; color: #8b949e; padding-left: 12px; border-left: 1px solid #30363d; margin: 2px 0;">{item.strip()}</div>'
        
        metrics_html += "</div>"

        # ปรับปรุงส่วน Insights (Explanations)
        insights_html = ""
        if explanations:
            items_content = ""
            for exp in explanations:
                # ทำความสะอาดข้อความ "ทำไม..." ออกเพื่อลดความซ้ำซ้อน
                display_text = exp.replace("ทำไม", "").replace("?", "").strip()
                if exp.startswith("ทำไม"):
                    items_content += f'<div style="font-weight: 700; color: #58a6ff; font-size: 0.95rem; margin-bottom: 4px;">{display_text}</div>'
                elif exp.startswith("  →"):
                    items_content += f'<div style="color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; margin-bottom: 8px;">{exp.strip().replace("→", "—")}</div>'
                elif "✓" in exp or "⚠" in exp or "ℹ" in exp:
                    items_content += f'<div style="font-size: 0.88rem; color: #8b949e; margin: 4px 0;">{exp}</div>'
                else:
                    items_content += f'<div style="font-size: 0.9rem; color: #e6edf3; margin: 4px 0;">{exp}</div>'

            insights_html = f'<div style="margin-top: 20px; border-top: 1px solid #30363d; padding-top: 16px;"><div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Data Insights & Rationale</div><div class="insight-bubble">{items_content}</div></div>'

        # เรนเดอร์ Card
        st.markdown(f'<div class="step-card"><div class="step-dot" style="border-color: {color}; color: {color}">{index + 1}</div><div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;"><span style="font-size: 1.1rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 1px;">{step_name}</span></div>{metrics_html}{other_items_html}{insights_html}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_comparison():
    comparison_data = get_comparison()
    if not comparison_data:
        st.info("ยังไม่มีข้อมูลการเปรียบเทียบ (ข้อมูลจะปรากฏขึ้นหากคุณกดย้อนกลับไปแก้ไขขั้นตอนก่อนหน้าแล้วทำใหม่จนเสร็จ)")
        return

    render_section_header(
        "Pipeline Comparison (Split View)",
        "เปรียบเทียบความแตกต่างแบบฝั่งซ้าย (เดิม) และฝั่งขวา (ใหม่)",
    )

    def flatten_dictionary(dictionary_data, prefix=""):
        items_dict = {}
        for key, value in dictionary_data.items():
            if isinstance(value, dict):
                items_dict.update(flatten_dictionary(value, prefix=f"{key}_"))
            else:
                items_dict[prefix + key] = value
        return items_dict

    for step_key in ["upload", "cleaning", "transformation", "model_process"]:
        if step_key not in comparison_data:
            continue
            
        step_data = comparison_data[step_key]
        previous_config = step_data.get("prev") or {}
        current_config = step_data.get("curr") or {}
        
        flat_previous = flatten_dictionary(previous_config)
        flat_current = flatten_dictionary(current_config)
        all_keys_list = sorted(list(set(flat_previous.keys()) | set(flat_current.keys())))

        # GitHub Header
        st.markdown(f"""
        <div style="background:#161B22; padding:8px 16px; border:1px solid #30363D; border-radius:6px 6px 0 0; margin-top:24px; border-bottom:1px solid #30363D;">
            <span style="color:#8B949E; font-family:monospace; font-size:0.85rem;">
                pipeline / <span style="color:#C9D1D9; font-weight:600;">{step_key}.cfg</span>
                <span style="margin-left:8px; background:#21262D; padding:2px 8px; border-radius:10px; font-size:0.75rem; color:#7D8590;">
                    {STEP_LABELS.get(step_key, step_key)}
                </span>
            </span>
        </div>
        <div style="border:1px solid #30363D; border-top:none; border-radius:0 0 6px 6px; overflow:hidden; background:#0D1117;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; border-bottom:1px solid #30363D;">
                <div style="padding:4px 12px; color:#8B949E; font-size:0.7rem; border-right:1px solid #30363D; background:#161B22; font-weight:600;">ORIGINAL CONFIG</div>
                <div style="padding:4px 12px; color:#8B949E; font-size:0.7rem; background:#161B22; font-weight:600;">CURRENT CONFIG</div>
            </div>
        """, unsafe_allow_html=True)
        
        diff_rows_html = []
        for key in all_keys_list:
            value_previous = flat_previous.get(key, "—")
            value_current = flat_current.get(key, "—")
            is_changed = str(value_previous) != str(value_current)
            
            left_background  = "background:rgba(248, 81, 73, 0.12);" if is_changed else "background:transparent;"
            right_background = "background:rgba(63, 185, 80, 0.15);" if is_changed else "background:transparent;"
            
            left_text_color  = "#F85149" if is_changed else "#8B949E"
            right_text_color = "#3FB950" if is_changed else "#C9D1D9"
            
            diff_rows_html.append(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; font-family:monospace; font-size:0.82rem; border-bottom:1px solid #21262D;">
                <div style="{left_background} color:{left_text_color}; padding:6px 12px; border-right:1px solid #30363D; word-break:break-all;">
                    <span style="opacity:0.5; margin-right:8px;">{"-" if is_changed else " "}</span>{key}: {value_previous}
                </div>
                <div style="{right_background} color:{right_text_color}; padding:6px 12px; word-break:break-all;">
                    <span style="opacity:0.5; margin-right:8px;">{"+" if is_changed else " "}</span>{key}: {value_current}
                </div>
            </div>
            """)
        
        st.markdown("".join(diff_rows_html), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear History", width="stretch"):
        clear_comparison()
        st.rerun()