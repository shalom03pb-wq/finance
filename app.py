import numpy as np, pandas as pd, yfinance as yf, streamlit as st
import plotly.graph_objects as go
from scipy import stats, optimize
from scipy.linalg import toeplitz, cholesky
import statsmodels.api as sm

st.set_page_config(page_title="KO · Modelos estocásticos", layout="wide")
TICKER = "KO"
PEERS = ["KO", "PEP", "KDP", "MNST", "PG", "MDLZ", "PM", "MO", "CL", "GIS", "KHC", "KMB"]
N = stats.norm.cdf

# ---------------------------------------------------------------- DATOS (yfinance)
@st.cache_data(ttl=3600, show_spinner="Descargando datos de Yahoo Finance…")
def load(period):
    tk = yf.Ticker(TICKER)
    h = tk.history(period=period, auto_adjust=False)
    h.index = h.index.tz_localize(None)
    d = tk.dividends
    d.index = d.index.tz_localize(None)
    try: info = tk.info
    except Exception: info = {}
    try: bs = tk.balance_sheet
    except Exception: bs = pd.DataFrame()
    try: news = tk.news
    except Exception: news = []
    def last(sym):
        try: return float(yf.Ticker(sym).history(period="5d")["Close"].dropna().iloc[-1])
        except Exception: return None
    spx = yf.Ticker("^GSPC").history(period=period)["Close"]; spx.index = spx.index.tz_localize(None)
    irx, fx = last("^IRX"), last("USDCOP=X")
    peers = yf.download(PEERS, period="2y", auto_adjust=True, progress=False)["Close"].dropna()
    return dict(h=h, div=d, info=info, bs=bs, news=news, spx=spx,
                rf=irx / 100 if irx else None, fx=fx, peers=peers, exps=list(tk.options))

@st.cache_data(ttl=900)
def chain(exp):
    oc = yf.Ticker(TICKER).option_chain(exp)
    return oc.calls, oc.puts

def fmt(v, kind="n"):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "n/d"
    if kind == "$": return f"${v:,.2f}"
    if kind == "B": return f"${v/1e9:,.1f} B"
    if kind == "%": return f"{v*100:.2f}%"
    return f"{v:,.2f}"

# ---------------------------------------------------------------- MODELOS
rng = np.random.default_rng

def to_paths(S0, incr):
    return S0 * np.exp(np.vstack([np.zeros((1, incr.shape[1])), np.cumsum(incr, 0)]))

def sim_gbm(S0, m, s, steps, n, g):
    return to_paths(S0, m + s * g.standard_normal((steps, n)))

def fit_ou(x):
    res = sm.OLS(np.diff(x), sm.add_constant(x[:-1])).fit()
    a, b = res.params
    b = min(b, 0.0)
    return dict(a=a, b=b, sig=res.resid.std(ddof=1), pval=res.pvalues[1])

def sim_ou(S0, p, steps, n, g):
    x = np.full(n, np.log(S0)); out = [x.copy()]
    for _ in range(steps):
        x = x + p["a"] + p["b"] * x + p["sig"] * g.standard_normal(n); out.append(x.copy())
    return np.exp(np.array(out))

def fit_jumps(r, k=3):
    mask = np.abs(r - r.mean()) > k * r.std()
    nj, j = r[~mask], r[mask]
    return dict(m=nj.mean(), s=nj.std(), lam=mask.mean(),
                muJ=j.mean() if len(j) else 0.0, sJ=j.std() if len(j) > 1 else 0.0, nj=int(mask.sum()))

def fit_merton_ml(r):
    def nll(t):
        m, ls, lg, mj, lsj = t
        s, lam, sj = np.exp(ls), 1 / (1 + np.exp(-lg)), np.exp(lsj)
        p = (1 - lam) * stats.norm.pdf(r, m, s) + lam * stats.norm.pdf(r, m + mj, np.sqrt(s**2 + sj**2))
        return -np.sum(np.log(p + 1e-300))
    x0 = [r.mean(), np.log(r.std() * .7), -3, 0, np.log(r.std() * 2)]
    o = optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=5000, xatol=1e-8, fatol=1e-8))
    m, ls, lg, mj, lsj = o.x
    return dict(m=m, s=np.exp(ls), lam=1 / (1 + np.exp(-lg)), muJ=mj, sJ=np.exp(lsj), nj=None, ll=-o.fun)

def sim_jump(S0, p, steps, n, g):
    k = g.poisson(p["lam"], (steps, n))
    inc = p["m"] + p["s"] * g.standard_normal((steps, n)) + k * p["muJ"] + np.sqrt(k) * p["sJ"] * g.standard_normal((steps, n))
    return to_paths(S0, inc)

def hurst(x, lags=range(2, 100)):
    lags = list(lags)
    tau = [np.std(x[l:] - x[:-l]) for l in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0], lags, tau

def sim_fbm(S0, m, s, H, steps, n, g):
    k = np.arange(steps)
    gam = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    L = cholesky(toeplitz(gam) + 1e-10 * np.eye(steps), lower=True)
    return to_paths(S0, m + s * (L @ g.standard_normal((steps, n))))

def sim_fractal(S0, r, steps, n, g, b=21):
    nb = int(np.ceil(steps / b))
    idx = g.integers(0, len(r) - b, (nb, n))[:, None, :] + np.arange(b)[None, :, None]
    return to_paths(S0, r[idx].reshape(nb * b, n)[:steps])

def bs(S, K, T, r, q, sig, cp):
    d1 = (np.log(S / K) + (r - q + .5 * sig**2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    if cp == "c": return S*np.exp(-q*T)*N(d1) - K*np.exp(-r*T)*N(d2), np.exp(-q*T)*N(d1)
    return K*np.exp(-r*T)*N(-d2) - S*np.exp(-q*T)*N(-d1), -np.exp(-q*T)*N(-d1)

def merton_struct(E, D, sE, r, T=1.0):
    def eq(v):
        V, sV = v
        d1 = (np.log(V / D) + (r + .5 * sV**2) * T) / (sV * np.sqrt(T)); d2 = d1 - sV * np.sqrt(T)
        return [V*N(d1) - D*np.exp(-r*T)*N(d2) - E, N(d1)*V*sV - sE*E]
    V, sV = optimize.fsolve(eq, [E + D, sE * E / (E + D)])
    d2 = (np.log(V / D) + (r - .5 * sV**2) * T) / (sV * np.sqrt(T))
    return dict(V=V, sV=sV, dd=d2, pd=N(-d2), D=D, E=E)

def sim_merton_struct(S0, p, r, steps, n, g):
    dt = 1 / 252
    V = p["V"] * np.exp(np.vstack([np.zeros((1, n)), np.cumsum((r - .5*p["sV"]**2)*dt + p["sV"]*np.sqrt(dt)*g.standard_normal((steps, n)), 0)]))
    d1 = (np.log(V / p["D"]) + (r + .5 * p["sV"]**2)) / p["sV"]; d2 = d1 - p["sV"]
    E = V * N(d1) - p["D"] * np.exp(-r) * N(d2)
    return S0 * E / E[0, 0]

@st.cache_data(show_spinner="Simulando modelos…")
def run_models(close, steps, n, seed, E, D, rf):
    g = rng(seed); lx = np.log(close); r = np.diff(lx); S0 = close[-1]
    m, s = r.mean(), r.std(ddof=1)
    P, par = {}, {}
    par["GBM"] = dict(m=m, s=s); P["GBM"] = sim_gbm(S0, m, s, steps, n, g)
    par["OU"] = fit_ou(lx); P["Ornstein-Uhlenbeck"] = sim_ou(S0, par["OU"], steps, n, g)
    par["Jump"] = fit_jumps(r); P["Saltos por difusión"] = sim_jump(S0, par["Jump"], steps, n, g)
    par["MertonML"] = fit_merton_ml(r); P["Merton (ML)"] = sim_jump(S0, par["MertonML"], steps, n, g)
    H, lags, tau = hurst(lx); H = float(np.clip(H, 0.05, 0.95)); par["H"] = (H, lags, tau)
    P["Fractal (bootstrap por bloques)"] = sim_fractal(S0, r, steps, n, g)
    P["Mov. browniano fraccional"] = sim_fbm(S0, m, s, H, steps, min(n, 1500), g)
    if E and D and rf is not None:
        sE = s * np.sqrt(252); par["Struct"] = merton_struct(E, D, sE, rf)
        P["Merton (1974 estructural)"] = sim_merton_struct(S0, par["Struct"], rf, steps, n, g)
    return P, par

def fan(paths, hist, title):
    last = hist.index[-1]; steps = paths.shape[0] - 1
    x = [last] + list(pd.bdate_range(last + pd.Timedelta(days=1), periods=steps))
    fig = go.Figure(); h = hist.iloc[-252:]
    fig.add_scatter(x=h.index, y=h.values, name="Histórico", line=dict(color="#444"))
    for i in range(min(25, paths.shape[1])):
        fig.add_scatter(x=x, y=paths[:, i], line=dict(width=.5, color="rgba(120,120,120,.25)"), showlegend=False, hoverinfo="skip")
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    fig.add_scatter(x=x, y=q[4], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=q[0], fill="tonexty", fillcolor="rgba(214,40,40,.12)", line=dict(width=0), name="P5–P95")
    fig.add_scatter(x=x, y=q[3], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=q[1], fill="tonexty", fillcolor="rgba(214,40,40,.25)", line=dict(width=0), name="P25–P75")
    fig.add_scatter(x=x, y=q[2], name="Mediana", line=dict(color="#d62828", width=2))
    fig.update_layout(title=title, height=430, yaxis_title="USD", margin=dict(t=50, b=10))
    return fig

def term_stats(p, S0):
    t = p[-1]
    return dict(media=t.mean(), mediana=np.median(t), P5=np.percentile(t, 5), P95=np.percentile(t, 95), prob_sube=(t > S0).mean())

# ---------------------------------------------------------------- SIDEBAR
st.title("🥤 The Coca-Cola Company (KO) · Modelos estocásticos")
with st.sidebar:
    st.header("Parámetros")
    period = st.selectbox("Histórico para calibrar", ["3y", "5y", "10y"], index=1)
    days = st.slider("Horizonte (días calendario)", 30, 730, 365, 5)
    nsim = st.select_slider("Simulaciones", [500, 1000, 2000, 5000], 2000)
    seed = st.number_input("Semilla", 0, 9999, 42)
    st.caption("Datos: Yahoo Finance vía yfinance. Educativo, no es asesoría financiera.")

D = load(period); h = D["h"]; close = h["Close"].dropna(); S0 = float(close.iloc[-1])
steps = max(5, round(days * 252 / 365)); Hy = days / 365
info, rf, fx = D["info"], D["rf"], D["fx"]
divs = D["div"]; q = float(divs[divs.index > divs.index[-1] - pd.Timedelta(days=365)].sum() / S0) if len(divs) else 0.0
E = info.get("marketCap")
Dbt = None
try: Dbt = float(D["bs"].loc["Total Debt"].dropna().iloc[0])
except Exception: pass
P, par = run_models(close.values, steps, nsim, int(seed), E, Dbt, rf)
sig_a = par["GBM"]["s"] * np.sqrt(252)

tabs = st.tabs(["Contexto", "GBM", "Ornstein-Uhlenbeck", "Saltos por difusión", "Merton 1974", "Merton ML",
                "Fractal", "fBm", "Black-Scholes", "Herding", "Strikes sugeridos", "KO vs CDT"])

def model_tab(name, expl, params=None):
    st.markdown(expl)
    if params: st.dataframe(pd.DataFrame(params, index=["valor"]).T.style.format("{:.6f}"), width=360)
    st.plotly_chart(fan(P[name], close, f"{name} — {days} días"), use_container_width=True)
    ts = term_stats(P[name], S0)
    c = st.columns(5)
    for col, (k, v) in zip(c, ts.items()):
        col.metric(k.replace("_", " "), f"{v*100:.1f}%" if k == "prob_sube" else f"${v:,.2f}")

# ---------------------------------------------------------------- CONTEXTO
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Precio (cierre)", fmt(S0, "$"), f"{(close.iloc[-1]/close.iloc[-2]-1)*100:.2f}% día")
    c[1].metric("Market cap", fmt(E, "B") if E else "n/d")
    c[2].metric("P/E (TTM) / Forward", f"{fmt(info.get('trailingPE'))} / {fmt(info.get('forwardPE'))}")
    c[3].metric("Dividend yield (12m)", fmt(q, "%"))
    c = st.columns(4)
    c[0].metric("Beta", fmt(info.get("beta")))
    c[1].metric("Rango 52 sem.", f"{fmt(info.get('fiftyTwoWeekLow'),'$')} – {fmt(info.get('fiftyTwoWeekHigh'),'$')}")
    c[2].metric("Vol. anualizada (hist.)", fmt(sig_a, "%"))
    c[3].metric("Tasa libre de riesgo (T-bill 13s)", fmt(rf, "%") if rf else "n/d")
    c = st.columns(4)
    c[0].metric("Margen neto", fmt(info.get("profitMargins"), "%") if info.get("profitMargins") else "n/d")
    c[1].metric("ROE", fmt(info.get("returnOnEquity"), "%") if info.get("returnOnEquity") else "n/d")
    c[2].metric("Deuda/Patrimonio", fmt(info.get("debtToEquity")))
    c[3].metric("USD/COP (Yahoo)", fmt(fx) if fx else "n/d")
    st.write(f"**{info.get('longName','KO')}** · {info.get('sector','')} · {info.get('industry','')}")
    st.caption(info.get("longBusinessSummary", ""))
    sma = go.Figure()
    sma.add_scatter(x=close.index, y=close, name="KO")
    sma.add_scatter(x=close.index, y=close.rolling(50).mean(), name="SMA 50")
    sma.add_scatter(x=close.index, y=close.rolling(200).mean(), name="SMA 200")
    sma.update_layout(title="Precio y medias móviles", height=380, margin=dict(t=50))
    st.plotly_chart(sma, use_container_width=True)
    both = pd.concat([close, D["spx"]], axis=1, keys=["KO", "S&P 500"]).dropna()
    cmp_ = go.Figure([go.Scatter(x=both.index, y=both[c] / both[c].iloc[0] * 100, name=c) for c in both])
    cmp_.update_layout(title="KO vs S&P 500 (base 100)", height=340, margin=dict(t=50))
    st.plotly_chart(cmp_, use_container_width=True)
    r = np.log(close).diff().dropna(); dd = (close / close.cummax() - 1).min()
    st.write(f"Retorno 1m: {(close.iloc[-1]/close.iloc[-22]-1)*100:.1f}% · 6m: {(close.iloc[-1]/close.iloc[-126]-1)*100:.1f}% · 1a: {(close.iloc[-1]/close.iloc[-252]-1)*100:.1f}% · Máx. drawdown ({period}): {dd*100:.1f}%")
    with st.expander("Noticias recientes"):
        for n_ in D["news"][:8]:
            ct = n_.get("content", n_)
            url = (ct.get("canonicalUrl") or {}).get("url") or n_.get("link")
            st.markdown(f"- [{ct.get('title','(sin título)')}]({url})" if url else f"- {ct.get('title','')}")

# ---------------------------------------------------------------- MODELOS
with tabs[1]:
    model_tab("GBM", "**Movimiento browniano geométrico:** dS = μS dt + σS dW. Parámetros: media y desviación de los log-retornos diarios históricos.", par["GBM"])
with tabs[2]:
    o = par["OU"]
    th = -o["b"]
    st.caption("Si la reversión no es significativa, el modelo se comporta casi como un paseo aleatorio.")
    model_tab("Ornstein-Uhlenbeck", "**OU sobre log-precio:** dX = θ(μ−X)dt + σdW, estimado por regresión AR(1).",
              dict(theta_diario=th, mu_log=(-o["a"] / o["b"]) if o["b"] < 0 else np.nan, sigma=o["sig"], pvalor_reversion=o["pval"]))
    if th > 0: st.info(f"Media de largo plazo implícita: ${np.exp(-o['a']/o['b']):,.2f} · vida media: {np.log(2)/th:,.0f} días")
with tabs[3]:
    j = par["Jump"]
    model_tab("Saltos por difusión", f"**Difusión con saltos:** se detectan saltos como retornos a más de 3σ ({j['nj']} observaciones); el resto es la parte difusiva. Los saltos llegan como Poisson(λ) con tamaño normal.",
              {k: v for k, v in j.items() if k != "nj"})
with tabs[4]:
    if "Struct" in par:
        sp = par["Struct"]
        c = st.columns(4)
        c[0].metric("Valor activos (V)", fmt(sp["V"], "B")); c[1].metric("Vol. activos", fmt(sp["sV"], "%"))
        c[2].metric("Distancia al default", fmt(sp["dd"])); c[3].metric("Prob. default 1a (neutral)", f"{sp['pd']*100:.6f}%")
        st.caption("Merton (1974): el patrimonio es una call sobre los activos con strike = deuda total (balance de Yahoo), vencimiento 1 año. Trayectorias bajo medida neutral al riesgo.")
        st.plotly_chart(fan(P["Merton (1974 estructural)"], close, "Merton estructural — patrimonio por acción"), use_container_width=True)
    else:
        st.warning("Yahoo no devolvió capitalización o deuda total para calibrar este modelo.")
with tabs[5]:
    mm = par["MertonML"]
    model_tab("Merton (ML)", "**Merton de saltos con máxima verosimilitud:** mezcla de normales (día normal vs. día con salto) estimada maximizando la log-verosimilitud con los retornos reales.",
              {k: v for k, v in mm.items() if k != "nj"})
with tabs[6]:
    H, lags, tau = par["H"]
    c = st.columns(3); c[0].metric("Hurst (H)", f"{H:.3f}"); c[1].metric("Dimensión fractal (2−H)", f"{2-H:.3f}")
    c[2].metric("Régimen", "Persistente" if H > .55 else "Antipersistente" if H < .45 else "≈ Aleatorio")
    lg = go.Figure(go.Scatter(x=np.log(lags), y=np.log(tau), mode="markers"))
    lg.update_layout(title="Escalamiento log-log (pendiente = H)", height=300, margin=dict(t=50)); st.plotly_chart(lg, use_container_width=True)
    model_tab("Fractal (bootstrap por bloques)", "**Fractal:** se estima H por escalamiento de la desviación de incrementos y se proyecta con *bootstrap* de bloques de 21 días, que preserva la estructura de memoria y clustering de volatilidad.")
with tabs[7]:
    H = par["H"][0]
    model_tab("Mov. browniano fraccional", f"**fBm** con H = {H:.3f} estimado de los datos: incrementos correlacionados (ruido gaussiano fraccional) simulados por Cholesky; H>0.5 persistencia, H<0.5 reversión.", dict(H=H, **par["GBM"]))

# ---------------------------------------------------------------- BLACK-SCHOLES + STRIKES
exps = D["exps"]
with tabs[8]:
    if not exps: st.warning("Sin cadena de opciones en Yahoo.")
    else:
        tgt = pd.Timestamp.today() + pd.Timedelta(days=days)
        default = int(np.argmin([abs((pd.Timestamp(e) - tgt).days) for e in exps]))
        exp = st.selectbox("Vencimiento", exps, index=default, key="exp")
        calls, puts = chain(exp); T = max((pd.Timestamp(exp) - pd.Timestamp.today()).days, 1) / 365
        st.write(f"S = {fmt(S0,'$')} · T = {T:.3f} años · r = {fmt(rf,'%')} · q = {fmt(q,'%')} · σ histórica = {fmt(sig_a,'%')}")
        near = calls[(calls.strike > S0 * .8) & (calls.strike < S0 * 1.2)].copy()
        near["BS σ hist"] = bs(S0, near.strike, T, rf or 0, q, sig_a, "c")[0]
        near["BS σ implícita"] = bs(S0, near.strike, T, rf or 0, q, near.impliedVolatility.clip(lower=.01), "c")[0]
        near["Mercado (mid)"] = (near.bid + near.ask) / 2
        st.dataframe(near[["strike", "bid", "ask", "Mercado (mid)", "BS σ hist", "BS σ implícita", "impliedVolatility", "openInterest"]].round(3), use_container_width=True)
        f = go.Figure(go.Scatter(x=near.strike, y=near.impliedVolatility, mode="lines+markers", name="IV calls"))
        pp = puts[(puts.strike > S0 * .8) & (puts.strike < S0 * 1.2)]
        f.add_scatter(x=pp.strike, y=pp.impliedVolatility, mode="lines+markers", name="IV puts")
        f.add_hline(y=sig_a, line_dash="dash", annotation_text="σ histórica")
        f.update_layout(title="Sonrisa de volatilidad implícita", height=340, margin=dict(t=50)); st.plotly_chart(f, use_container_width=True)

with tabs[10]:
    if exps:
        st.caption(f"Cadena del vencimiento {exp}. Deltas calculadas con Black-Scholes usando la IV de mercado de cada strike; solo strikes con open interest > 0.")
        c_ = calls[(calls.impliedVolatility > .01) & (calls.openInterest > 0)].copy()
        p_ = puts[(puts.impliedVolatility > .01) & (puts.openInterest > 0)].copy()
        r_ = rf or 0
        c_["delta"] = bs(S0, c_.strike, T, r_, q, c_.impliedVolatility, "c")[1]
        p_["delta"] = bs(S0, p_.strike, T, r_, q, p_.impliedVolatility, "p")[1]
        near_k = lambda df, k: df.iloc[(df.strike - k).abs().argmin()]
        near_d = lambda df, d: df.iloc[(df.delta - d).abs().argmin()]
        up, dn = S0 * np.exp(sig_a * np.sqrt(T)), S0 * np.exp(-sig_a * np.sqrt(T))
        sel = [("Call ATM", "Call", near_k(c_, S0)), ("Venta call cubierta (Δ≈0.30)", "Call", near_d(c_, .30)),
               ("Put ATM", "Put", near_k(p_, S0)), ("Put vendida con caja (Δ≈−0.30)", "Put", near_d(p_, -.30)),
               ("Protección put (Δ≈−0.20)", "Put", near_d(p_, -.20)),
               (f"Techo +1σ (${up:,.2f})", "Call", near_k(c_, up)), (f"Piso −1σ (${dn:,.2f})", "Put", near_k(p_, dn))]
        tab = pd.DataFrame([dict(Estrategia=n_, Tipo=t, Strike=r.strike, Bid=r.bid, Ask=r.ask, Último=r.lastPrice,
                                 IV=r.impliedVolatility, Delta=r.delta, OpenInt=r.openInterest, Volumen=r.volume) for n_, t, r in sel])
        st.dataframe(tab.style.format({"Strike": "{:.2f}", "Bid": "{:.2f}", "Ask": "{:.2f}", "Último": "{:.2f}", "IV": "{:.1%}", "Delta": "{:.2f}"}), use_container_width=True)
        st.caption("Los strikes salen de la cadena real de Yahoo; la selección por delta y por ±1σ es un criterio de orientación, no una recomendación.")

# ---------------------------------------------------------------- HERDING
with tabs[9]:
    px = D["peers"]; rr = px.pct_change().dropna() * 100
    rm = rr.mean(1); csad = rr.sub(rm, axis=0).abs().mean(1)
    X = sm.add_constant(pd.DataFrame({"|Rm|": rm.abs(), "Rm²": rm**2}))
    res = sm.OLS(csad, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    g2, p2 = res.params["Rm²"], res.pvalues["Rm²"]
    st.markdown("**Herding (CSAD, Chang-Cheong-Lee 2000):** CSAD = γ0 + γ1|Rm| + γ2 Rm². Un γ2 **negativo y significativo** indica que, en movimientos extremos, los activos se parecen más entre sí (comportamiento de manada). Cross-section: KO y pares del sector de consumo básico, 2 años diarios.")
    c = st.columns(3); c[0].metric("γ2", f"{g2:.4f}"); c[1].metric("p-valor (HAC)", f"{p2:.4f}")
    c[2].metric("Conclusión", "Herding" if (g2 < 0 and p2 < .05) else "Sin evidencia")
    xs = np.linspace(rm.min(), rm.max(), 100)
    fh = go.Figure(go.Scatter(x=rm, y=csad, mode="markers", name="Días", marker=dict(size=4, opacity=.5)))
    fh.add_scatter(x=xs, y=res.params["const"] + res.params["|Rm|"] * np.abs(xs) + g2 * xs**2, name="Ajuste", line=dict(color="red"))
    fh.update_layout(xaxis_title="Rm (%)", yaxis_title="CSAD", height=380, margin=dict(t=30)); st.plotly_chart(fh, use_container_width=True)
    st.write(f"Desviación diaria promedio de KO vs. sector: {(rr['KO']-rm).abs().mean():.3f} pp · Peers: {', '.join(px.columns)}")

# ---------------------------------------------------------------- CDT
with tabs[11]:
    st.subheader("¿Y si el mismo dinero va a un CDT?")
    c = st.columns(4)
    inv = c[0].number_input("Monto a invertir (COP)", 1_000_000, 10_000_000_000, 10_000_000, 1_000_000)
    ea = c[1].number_input("Tasa CDT E.A. (%)", 0.0, 40.0, 9.0, 0.1, help="Valor de ejemplo editable: ingresa la tasa vigente de tu banco para el plazo elegido.")
    ret = c[2].number_input("Retención en la fuente (%)", 0.0, 20.0, 0.0, 0.5)
    fxc = c[3].number_input("Variación USD/COP esperada (%)", -30.0, 30.0, 0.0, 0.5, help="0 = tasa de cambio constante.")
    cdt_gain = inv * ((1 + ea / 100) ** Hy - 1) * (1 - ret / 100); cdt_final = inv + cdt_gain
    st.metric(f"CDT a {days} días", f"${cdt_final:,.0f} COP", f"+${cdt_gain:,.0f}")
    st.caption(f"Equivale a USD {inv/fx:,.0f} al cambio {fx:,.0f} COP/USD (Yahoo). KO = valor final del precio simulado + dividendos estimados al yield de 12m (sin reinvertir), en COP." if fx else "USD/COP no disponible.")
    rows = []
    for nm, p in P.items():
        val = inv * (p[-1] / S0 + q * Hy) * (1 + fxc / 100)
        rows.append({"Modelo": nm, "KO mediana": np.median(val), "KO P5": np.percentile(val, 5), "KO P95": np.percentile(val, 95),
                     "Prob. KO > CDT": (val > cdt_final).mean(), "Prob. perder capital": (val < inv).mean()})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"KO mediana": "${:,.0f}", "KO P5": "${:,.0f}", "KO P95": "${:,.0f}", "Prob. KO > CDT": "{:.1%}", "Prob. perder capital": "{:.1%}"}), use_container_width=True)
    b = go.Figure(go.Bar(x=df["Modelo"], y=df["KO mediana"], name="KO (mediana)"))
    b.add_hline(y=cdt_final, line_color="green", annotation_text="CDT"); b.add_hline(y=inv, line_dash="dot", annotation_text="Capital")
    b.update_layout(height=380, yaxis_title="COP", margin=dict(t=30)); st.plotly_chart(b, use_container_width=True)
    st.warning("El CDT tiene retorno pactado y el riesgo de KO es de mercado y cambiario. Esta comparación es ilustrativa, no asesoría financiera.")
