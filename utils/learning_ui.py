import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.nn_helpers import A, C, G, GRID, MUTED, R, TEXT, plotly_layout
from utils.voice import render_voice_button


def _accent_bar(color):
    st.markdown(
        f"""
        <div style="height:4px; width:84px; border-radius:999px; background:{color}; margin:0 0 14px 0;"></div>
        """,
        unsafe_allow_html=True,
    )


def render_learning_journey(title, intro, bullets, analogy, audio_text="", accent=C, key_suffix=""):
    with st.container(border=True):
        left, right = st.columns([1.7, 1.1])
        with left:
            _accent_bar(accent)
            st.caption("BEGINNER LEARNING TRACK")
            st.subheader(title)
            st.write(intro)
        with right:
            st.caption("WHAT TO NOTICE")
            for item in bullets:
                st.markdown(f"- {item}")

        st.caption("HUMAN ANALOGY")
        st.info(analogy)

    if audio_text:
        render_voice_button(audio_text, key_suffix=f"journey_{key_suffix}")


def render_step_grid(steps, columns=3):
    if not steps:
        return

    col_count = max(1, min(columns, len(steps)))
    cols = st.columns(col_count)
    for idx, step in enumerate(steps):
        accent = step.get("accent", C)
        with cols[idx % col_count]:
            with st.container(border=True):
                _accent_bar(accent)
                st.caption(step.get("eyebrow", "LIVE STEP").upper())
                st.markdown(f"#### {step.get('title', '')}")
                st.markdown(
                    f"""
                    <div style="font-family:'JetBrains Mono', monospace; font-size:24px; color:{accent}; font-weight:700; margin: 6px 0 10px 0;">
                        {step.get("value", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(step.get("caption", ""))


def render_ai_coach_panel(title, narratives, key_suffix="", accent=C):
    if not narratives:
        return None

    labels = list(narratives.keys())
    coach_focus = st.selectbox(
        title,
        labels,
        key=f"coach_focus_{key_suffix}",
    )
    coach_text = narratives[coach_focus]

    with st.container(border=True):
        _accent_bar(accent)
        st.caption("AI COACH")
        st.markdown(f"#### {coach_focus}")
        st.write(coach_text)

    render_voice_button(coach_text, key_suffix=f"coach_{key_suffix}")
    return coach_focus, coach_text


def render_story_card(title, body, eyebrow="GUIDED STORY", accent=C, key_suffix="", voice_text=None):
    with st.container(border=True):
        _accent_bar(accent)
        st.caption(eyebrow.upper())
        st.markdown(f"#### {title}")
        st.write(body)

    spoken_text = voice_text if voice_text is not None else body
    if key_suffix and spoken_text:
        render_voice_button(spoken_text, key_suffix=key_suffix)


def render_summary_panel(title, bullets, eyebrow="LEARNING SNAPSHOT", accent=C):
    if not bullets:
        return

    with st.container(border=True):
        _accent_bar(accent)
        st.caption(eyebrow.upper())
        st.markdown(f"#### {title}")
        for item in bullets:
            st.markdown(f"- {item}")


def render_visualization_mode(module_key, accent=C, subject="this module"):
    descriptions = {
        "Friendly Dashboard": f"Shows the clearest core metrics and 2D charts for {subject}.",
        "Immersive Coach": f"Puts the teaching layer first so {subject} feels guided and step-by-step.",
        "3D Visualization Explorer": f"Unlocks the 3D views for {subject} and makes the geometry more interactive.",
    }
    with st.container(border=True):
        _accent_bar(accent)
        st.caption("VISUALIZATION MODE")
        mode = st.radio(
            "Visualization mode",
            list(descriptions.keys()),
            horizontal=True,
            key=f"viz_mode_{module_key}",
            label_visibility="collapsed",
        )
        st.markdown(f"#### {mode}")
        st.write(descriptions[mode])
    return mode


def heatmap_with_text(
    z,
    x_labels,
    y_labels,
    title,
    zmid=0.0,
    colorscale=None,
    height=320,
    colorbar_title="Value",
):
    arr = np.array(z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if colorscale is None:
        colorscale = [[0.0, "#DC2626"], [0.5, "#F8FAFC"], [1.0, "#2563EB"]]

    text = [[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in arr]
    fig = go.Figure(
        go.Heatmap(
            z=arr,
            x=x_labels,
            y=y_labels,
            zmid=zmid,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, tickfont=dict(color=TEXT)),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, family="Montserrat", size=18)),
        **plotly_layout(
            height=height,
            xaxis=dict(color=MUTED, gridcolor=GRID),
            yaxis=dict(color=MUTED, gridcolor=GRID),
            margin=dict(l=40, r=20, t=55, b=35),
        ),
    )
    return fig


def contribution_bar(
    labels,
    values,
    title,
    positive=C,
    negative=R,
    neutral=A,
    height=280,
    y_title="Contribution",
):
    colors = []
    for val in values:
        if val > 1e-12:
            colors.append(positive)
        elif val < -1e-12:
            colors.append(negative)
        else:
            colors.append(neutral)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, family="Montserrat", size=18)),
        **plotly_layout(
            height=height,
            yaxis=dict(title=y_title, color=MUTED, gridcolor=GRID),
            xaxis=dict(color=MUTED, gridcolor=GRID),
            margin=dict(l=40, r=20, t=55, b=35),
        ),
    )
    return fig


def line_story_chart(series, title, y_title, height=300):
    fig = go.Figure()
    for item in series:
        fig.add_trace(
            go.Scatter(
                x=item["x"],
                y=item["y"],
                mode="lines+markers",
                name=item["name"],
                line=dict(color=item.get("color", C), width=3, dash=item.get("dash", "solid")),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, family="Montserrat", size=18)),
        **plotly_layout(
            height=height,
            yaxis=dict(title=y_title, color=MUTED, gridcolor=GRID),
            xaxis=dict(title="Step", color=MUTED, gridcolor=GRID),
            margin=dict(l=40, r=20, t=55, b=35),
        ),
    )
    return fig


def scatter3d_story(points, title, x_title, y_title, z_title, height=420):
    fig = go.Figure()
    for item in points:
        fig.add_trace(
            go.Scatter3d(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                mode=item.get("mode", "lines+markers"),
                name=item["name"],
                marker=dict(
                    size=item.get("size", 5),
                    color=item.get("color_values", item.get("color", C)),
                    colorscale=item.get("colorscale", "Turbo"),
                    opacity=0.9,
                ),
                line=dict(color=item.get("line_color", item.get("color", C)), width=item.get("line_width", 5)),
                text=item.get("text"),
                hovertemplate=item.get("hovertemplate", "%{text}<extra></extra>"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, family="Montserrat", size=18)),
        scene=dict(
            xaxis=dict(title=x_title, color=MUTED, gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title=y_title, color=MUTED, gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title=z_title, color=MUTED, gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        **plotly_layout(height=height, margin=dict(l=0, r=0, t=55, b=0)),
    )
    return fig
