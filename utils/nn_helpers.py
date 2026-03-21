import numpy as np
import plotly.graph_objects as go

# ──── SUPERHERO COMIC COLOR PALETTE ────
BG = "#020617"; SURF = "#0F172A"; CARD = "#020617"
P = "#EF4444" # Flash Red
C = "#3B82F6" # Superman Blue
G = "#22C55E" # Hulk Green
A = "#FACC15" # Wolverine Yellow
R = "#DC2626" # Danger Red
TEXT = "#FFFFFF" # White
MUTED = "#94A3B8" # Slate
GRID = "#334155" # Dark Grid
LAYER_COLS = ["#EF4444", "#3B82F6", "#FACC15", "#22C55E", "#A855F7", "#F97316", "#3B82F6"]

# ──── PLOTLY BASE CONFIG ────
# Separated to avoid 'multiple values for keyword' errors in update_layout
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="'Bangers', cursive"),
    margin=dict(t=50, b=40, l=50, r=20)
)

PLOTLY_AXIS = dict(
    gridcolor=GRID, 
    zerolinecolor=GRID, 
    color=MUTED
)

def plotly_layout(**kwargs):
    """Safely merges base layout, axis defaults, and custom overrides."""
    base = dict(**PLOTLY_BASE)
    # Apply axis defaults if not overridden
    if 'xaxis' not in kwargs: base['xaxis'] = PLOTLY_AXIS
    if 'yaxis' not in kwargs: base['yaxis'] = PLOTLY_AXIS
    base.update(kwargs)
    return base

# ──── MATH HELPERS ────
ACTS = {
    "Sigmoid":   {"fn": lambda z: 1/(1+np.exp(-np.clip(z,-500,500))), "d": lambda a: a*(1-a),          "eq": "σ(z)=1/(1+e⁻ᶻ)"},
    "ReLU":      {"fn": lambda z: np.maximum(0,z),                    "d": lambda a: (a>0).astype(float),"eq": "max(0,z)"},
    "Tanh":      {"fn": lambda z: np.tanh(z),                         "d": lambda a: 1-a**2,            "eq": "tanh(z)"},
    "Linear":    {"fn": lambda z: z,                                   "d": lambda a: np.ones_like(a),   "eq": "f(z)=z"},
    "Leaky ReLU":{"fn": lambda z: np.where(z>=0,z,0.01*z),            "d": lambda a: np.where(a>=0,1,0.01),"eq": "z≥0?z:0.01z"},
    "Swish":     {"fn": lambda z: z/(1+np.exp(-np.clip(z,-500,500))), "d": lambda z: (lambda s: s + z*s*(1-s))(1/(1+np.exp(-np.clip(z,-500,500)))), "eq": "z·σ(z)"},
}

LOSSES = {
    "MSE":   {"fn": lambda p,t: 0.5*np.mean((p-t)**2),              "d": lambda p,t: p-t,                    "eq": "0.5·(ŷ−y)²"},
    "BCE":   {"fn": lambda p,t: -np.mean(t*np.log(np.clip(p,1e-9,1))+(1-t)*np.log(np.clip(1-p,1e-9,1))),
              "d":  lambda p,t: -(t/np.clip(p,1e-9,1))+(1-t)/np.clip(1-p,1e-9,1),                           "eq": "−(y·logŷ+(1−y)·log(1−ŷ))"},
    "MAE":   {"fn": lambda p,t: np.mean(np.abs(p-t)),               "d": lambda p,t: np.sign(p-t),           "eq": "|ŷ−y|"},
    "Huber": {"fn": lambda p,t: np.mean(np.where(np.abs(p-t)<=1, 0.5*(p-t)**2, np.abs(p-t)-0.5)),
              "d":  lambda p,t: np.where(np.abs(p-t)<=1, p-t, np.sign(p-t)),                                 "eq": "δ² if|δ|≤1 else |δ|−0.5"},
}

def xe_init(n): return np.random.uniform(-1,1) * np.sqrt(2/max(n,1))

def make_weights(n_in, hidden_sizes):
    ws=[]; s=n_in
    for h in hidden_sizes:
        ws.append((np.array([[xe_init(s) for _ in range(s)] for _ in range(h)]),
                   np.array([[xe_init(s)] for _ in range(h)])));  s=h
    ws.append((np.array([[xe_init(s) for _ in range(s)]]),
               np.array([[xe_init(s)]])))
    return ws

def forward_pass(X, weights, h_acts, o_act):
    Zs=[]; As=[X]; A=X
    for li,(W,b) in enumerate(weights):
        Z=W@A+b; Zs.append(Z)
        A=ACTS[o_act if li==len(weights)-1 else h_acts[li]]["fn"](Z); As.append(A)
    return Zs, As

def backward_pass(weights, As, y_true, h_acts, o_act, loss_fn):
    NL=len(weights); grads=[None]*NL
    dLdA=LOSSES[loss_fn]["d"](As[-1], np.array([[y_true]]))
    dAdZ=ACTS[o_act]["d"](As[-1]); dLdZ=dLdA*dAdZ
    grads[-1]=dict(dLdA=dLdA,dAdZ=dAdZ,dLdZ=dLdZ,
                   dLdW=dLdZ@As[-2].T, dLdb=dLdZ.copy(),
                   dLdAp=weights[-1][0].T@dLdZ)
    cur=grads[-1]["dLdAp"]
    for li in range(NL-2,-1,-1):
        act=h_acts[li]; dAdZ=ACTS[act]["d"](As[li+1])
        dLdZ=cur*dAdZ; dLdAp=weights[li][0].T@dLdZ
        grads[li]=dict(dLdA=cur,dAdZ=dAdZ,dLdZ=dLdZ,
                       dLdW=dLdZ@As[li].T, dLdb=dLdZ.copy(), dLdAp=dLdAp)
        cur=dLdAp
    return grads

def hex2rgba(h, a):
    h = h.lstrip('#')
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

def draw_network(sizes, labels, vals=None, highlight=None):
    n=len(sizes); mx=max(sizes)
    ns=max(14,min(30,int(130/max(mx,1)))); fs=max(6,min(9,int(ns*0.27)))
    H=max(260,mx*(ns+14)+80); W_f=max(360,n*130)
    lx=np.linspace(0.06,0.94,n).tolist() if n>1 else [0.5]
    SHOW=6

    def ny(tot,i): return np.linspace(0.87,0.13,tot)[i] if tot>1 else 0.5

    fig=go.Figure()
    positions=[]
    for li,(x,nn) in enumerate(zip(lx,sizes)):
        pos=[]
        if nn<=SHOW:
            for i in range(nn): pos.append((x,ny(nn,i),i,False))
        else:
            for i in range(3): pos.append((x,ny(SHOW+2,i),i,False))
            pos.append((x,ny(SHOW+2,3),-1,True))
            pos.append((x,ny(SHOW+2,4),nn-1,False))
        positions.append(pos)

    for li in range(n-1):
        for (x0,y0,_,e0) in positions[li]:
            for (x1,y1,_,e1) in positions[li+1]:
                if e0 or e1: continue
                # Line col fixed to P (Red)
                line_col = hex2rgba(P, 0.45)
                fig.add_shape(type="line",x0=x0,y0=y0,x1=x1,y1=y1,
                    xref="paper",yref="paper",layer="below",
                    line=dict(color=line_col, width=1.0))

    for li,pos in enumerate(positions):
        col=LAYER_COLS[li%len(LAYER_COLS)]
        for (nx,ny_,ni,ellip) in pos:
            if ellip:
                fig.add_trace(go.Scatter(x=[nx],y=[ny_],mode="text",text=["⋮"],
                    textfont=dict(size=18,color=MUTED),showlegend=False,hoverinfo="skip"))
                continue
            if li==0: lbl=f"x{ni+1}"
            elif li==n-1: lbl="ŷ"
            else: lbl=f"h{ni+1}"
            val_s=""
            if vals and li<len(vals) and ni<len(vals[li]):
                v=vals[li][ni]; val_s=f"\n{v:.3f}"
                col=G if v>0.5 else R if v<-0.5 else col
            
            is_hl = highlight is not None and li==highlight
            marker_col = hex2rgba(col, 1.0 if is_hl else 0.8)
            fig.add_trace(go.Scatter(x=[nx],y=[ny_],mode="markers+text",
                marker=dict(size=ns,color=marker_col,
                    line=dict(color="#F8FAFC",width=2 if is_hl else 1)),
                text=[f"{lbl}{val_s}"],textposition="middle center",
                textfont=dict(size=fs,color="#FFF" if not ellip else TEXT,family="Inter"),
                showlegend=False,hoverinfo="none"))
    for x,lbl in zip(lx,labels):
        fig.add_annotation(x=x,y=-0.06,text=f"<b>{lbl}</b>",showarrow=False,
            font=dict(size=10,color=TEXT),xanchor="center",xref="paper",yref="paper")
    
    fig.update_layout(height=H,xaxis=dict(visible=False,range=[-0.02,1.02]),
        yaxis=dict(visible=False,range=[-0.1,1.05]),
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=50))
    return fig

def loss_chart(losses, title="Training Loss", col=None):
    col = col or P
    fig=go.Figure()
    x=list(range(1,len(losses)+1))
    fig.add_trace(go.Scatter(x=x,y=losses,mode="lines",
        line=dict(color=col,width=2.5),
        fill="tozeroy",fillcolor=f"{col}18",name="Loss"))
    fig.update_layout(
        title=dict(text=title,font=dict(color=TEXT,size=14,family="Inter")),
        **plotly_layout())
    return fig
