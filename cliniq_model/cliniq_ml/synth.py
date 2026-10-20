"""Realistic *noisy* synthetic multi-site cohort.

Why this exists and what it is NOT
----------------------------------
The original generator drew each class from clean, non-overlapping threshold bands, so 98.5% accuracy
was circular. This generator is deliberately hard:

* labels come from a latent disease process, NOT from analyte cut-offs
* continuous severity -> mild disease overlaps the healthy tail
* comorbidity (diabetes -> CKD -> anemia ...), age/sex effects
* mimics ("traps"): dehydration, stress leukocytosis, pre-diabetes, controlled diabetics,
  non-thyroidal illness, subclinical thyroid disease, mixed anemia morphology
* informative missingness (tests are ordered when disease is suspected)
* per-lab analyser bias + noise, one lab (site 5) with a stronger shift for external validation
* ground-truth label noise

It is still synthetic. Numbers from it are a regression/engineering benchmark, never clinical evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BOUNDS, LABELS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _site_params(rng: np.random.Generator, site: int, shifted: bool) -> dict:
    s = 2.5 if shifted else 1.0
    return dict(
        hb=rng.normal(0, 0.10) * s - (0.3 if shifted else 0),
        glu=rng.normal(0, 2.5) * s + (5.0 if shifted else 0),
        creat=rng.normal(0, 0.03) * s + (0.12 if shifted else 0),
        chol=rng.normal(0, 4.0) * s,
        tsh=np.exp(rng.normal(0, 0.07) * s + (0.18 if shifted else 0)),
        noise=1.0 + 0.15 * rng.random() + (0.25 if shifted else 0),
        age_shift=8.0 if shifted else rng.normal(0, 2.0),
        miss=(0.75 if shifted else rng.uniform(0.9, 1.1)),
    )


def generate_site(n: int, site: int, rng: np.random.Generator, shifted: bool = False) -> pd.DataFrame:
    P = _site_params(rng, site, shifted)
    nz = P["noise"]

    age = np.clip(rng.normal(45 + P["age_shift"], 16, n), 18, 92)
    male = (rng.random(n) < 0.48).astype(int)
    fem = 1 - male
    a = (age - 45) / 10.0

    # ---- latent disease flags with comorbidity ------------------------------------------------
    diab = rng.random(n) < _sigmoid(-2.25 + 0.45 * a + 0.1 * male)
    chol = rng.random(n) < _sigmoid(-1.55 + 0.25 * a + 0.75 * diab - 0.1 * fem)
    ckd = rng.random(n) < _sigmoid(-3.35 + 0.55 * a + 1.0 * diab)
    anem = rng.random(n) < _sigmoid(-2.0 + 0.55 * fem + 1.5 * ckd + 0.1 * a)
    hypo = rng.random(n) < _sigmoid(-3.4 + 1.0 * fem + 0.35 * a)
    hyper = (rng.random(n) < _sigmoid(-4.0 + 0.5 * fem)) & ~hypo
    inf = rng.random(n) < _sigmoid(-2.4 + 0.4 * diab)

    sev = {k: rng.beta(1.6, 2.4, n) for k in LABELS}

    def noise(sd, size=n):
        return rng.normal(0, sd * nz, size)

    # ---- haematology --------------------------------------------------------------------------
    hb_base = np.where(male == 1, 14.8, 13.3) + P["hb"]
    hb = hb_base + noise(1.05)
    hb = hb - anem * (0.9 + 4.6 * sev["anemia"]) - ckd * 0.7 * sev["ckd"] - inf * 0.25
    morph = rng.choice(3, n, p=[0.55, 0.15, 0.30])           # iron-def / macro / normocytic
    mcv = 88 + noise(4.5)
    mcv = mcv + anem * ((morph == 0) * -(6 + 18 * sev["anemia"]) + (morph == 1) * (6 + 14 * sev["anemia"]))
    hct = 3.0 * hb + noise(1.0)
    rbc = (hct * 10.0 / mcv) * (1 + noise(0.025))
    mch = hb * 10.0 / rbc * (1 + noise(0.015))
    mchc = hb / hct * 100.0 * (1 + noise(0.015))

    stress = (~inf) & (rng.random(n) < 0.08)                 # leukocytosis mimic
    inf_type = rng.choice(3, n, p=[0.60, 0.30, 0.10])        # bacterial / viral / other
    bact, vir = inf & (inf_type == 0), inf & (inf_type == 1)
    wbc = 7.0 + noise(1.6)
    wbc = wbc + bact * (3 + 12 * sev["infection"]) - vir * (0.5 + 3.0 * sev["infection"]) + stress * 2.5
    wbc = np.clip(wbc, 0.8, None)
    neut = 58 + noise(7.5) + bact * (10 + 22 * sev["infection"]) - vir * (10 + 15 * sev["infection"]) + stress * 6
    neut = np.clip(neut, 5, 96)
    lymph = np.clip(100 - neut - np.clip(rng.normal(8, 2.5, n), 2, 16), 2, 90)
    plt = np.exp(np.log(265) + rng.normal(0, 0.21 * nz, n))
    dengue = vir & (rng.random(n) < 0.5)
    plt = plt * np.where(dengue, np.clip(0.78 - 0.65 * sev["infection"], 0.06, 1), 1.0) * np.where(bact, 1.1, 1.0)

    # ---- glycaemia ----------------------------------------------------------------------------
    prediab = (~diab) & (rng.random(n) < 0.18)
    controlled = diab & (rng.random(n) < 0.30)
    fbs = 91 + P["glu"] + noise(8.5) + prediab * (14 + noise(5))
    fbs = np.where(diab & ~controlled, 112 + P["glu"] + sev["diabetes"] * 185 + noise(12), fbs)
    fbs = np.where(controlled, 111 + P["glu"] + noise(10), fbs)
    hba1c = 5.3 + noise(0.32) + prediab * 0.55
    hba1c = np.where(diab, 6.6 + sev["diabetes"] * 3.4 + noise(0.45), hba1c)
    hba1c = np.where(controlled, 6.9 + noise(0.35), hba1c)

    # ---- lipids -------------------------------------------------------------------------------
    borderline = (~chol) & (rng.random(n) < 0.25)
    tc = 183 + P["chol"] + noise(27) + borderline * 30
    tc = np.where(chol, 215 + P["chol"] + sev["hypercholesterolemia"] * 95 + noise(24), tc)
    hdl = np.where(male == 1, 46, 55) + noise(8.5) - diab * 5
    tg = np.exp(np.log(125) + rng.normal(0, 0.45 * nz, n)) * (1 + 0.25 * diab + 0.3 * chol)
    ldl = tc - hdl - tg / 5.0 + noise(9)
    tc, hdl = np.clip(tc, 80, None), np.clip(hdl, 15, None)

    # ---- renal / liver ------------------------------------------------------------------------
    dehyd = (~ckd) & (rng.random(n) < 0.05)                  # prerenal mimic
    creat = np.where(male == 1, 0.95, 0.76) + P["creat"] + noise(0.13) + diab * 0.02
    creat = creat * np.where(ckd, 1.30 + 3.2 * sev["ckd"], 1.0) * np.where(dehyd, 1.25, 1.0)
    creat = np.clip(creat, 0.35, None)
    urea = (27 + noise(6.5)) * np.where(ckd, 1.25 + 2.0 * sev["ckd"], 1.0) * np.where(dehyd, 1.5, 1.0)
    fatty = (rng.random(n) < 0.12 + 0.18 * diab + 0.10 * chol)
    alt = np.exp(np.log(24) + rng.normal(0, 0.42 * nz, n)) * np.where(fatty, 1.6, 1.0) * np.where(vir, 1 + 3 * sev["infection"], 1.0)
    ast = np.exp(np.log(23) + rng.normal(0, 0.38 * nz, n)) * np.where(fatty, 1.35, 1.0) * np.where(vir, 1 + 2.5 * sev["infection"], 1.0)

    # ---- thyroid ------------------------------------------------------------------------------
    tsh = np.exp(np.log(1.8) + rng.normal(0, 0.45 * nz, n)) * P["tsh"]
    ft4 = 1.2 + noise(0.17)
    sub = rng.random(n) < 0.70
    tsh_hypo = np.where(sub, 4.6 + sev["hypothyroidism"] * 6 + np.abs(noise(0.9)), 10 + np.exp(sev["hypothyroidism"] * 3) * 6)
    ft4_hypo = np.where(sub, 1.0 + noise(0.17), 0.62 + noise(0.12))
    sub2 = rng.random(n) < 0.40
    tsh_hyper = np.where(sub2, np.exp(np.log(0.17) + rng.normal(0, 0.45, n)), np.exp(np.log(0.02) + rng.normal(0, 0.8, n)))
    ft4_hyper = np.where(sub2, 1.5 + noise(0.18), 2.4 + np.abs(noise(0.45)))
    tsh = np.where(hypo, tsh_hypo, np.where(hyper, tsh_hyper, tsh))
    ft4 = np.where(hypo, ft4_hypo, np.where(hyper, ft4_hyper, ft4))
    nti = inf & (rng.random(n) < 0.30) & ~hypo & ~hyper       # non-thyroidal illness suppresses TSH
    tsh = np.where(nti, tsh * 0.55, tsh)

    df = pd.DataFrame(dict(
        age=age.round(0), sex=male, hb=hb, rbc=rbc, hct=hct, mcv=mcv, mch=mch, mchc=mchc,
        wbc=wbc, neut_pct=neut, lymph_pct=lymph, platelets=plt, fbs=fbs, hba1c=hba1c,
        total_chol=tc, ldl=ldl, hdl=hdl, tg=tg, creatinine=creat, urea=urea, alt=alt, ast=ast,
        tsh=tsh, ft4=ft4,
    ))
    # lab-style rounding
    dec = dict(hb=1, rbc=2, hct=1, mcv=1, mch=1, mchc=1, wbc=1, neut_pct=0, lymph_pct=0, platelets=0, fbs=0,
               hba1c=1, total_chol=0, ldl=0, hdl=0, tg=0, creatinine=2, urea=0, alt=0, ast=0, tsh=2, ft4=2)
    for c, d in dec.items():
        df[c] = df[c].round(d)
    for c, (lo, hi) in BOUNDS.items():
        if c in df and c not in ("age",):
            df[c] = df[c].clip(lo, hi)

    # ---- informative missingness --------------------------------------------------------------
    def present(base, boost_mask=None, boost=0.0):
        p = np.full(n, base * P["miss"])
        if boost_mask is not None:
            p = np.where(boost_mask, np.minimum(0.97, p + boost), p)
        return rng.random(n) < np.clip(p, 0, 0.99)

    m_diff = present(0.88, inf, 0.1)
    m_lip = present(0.60, diab | chol, 0.2)
    m_ren = present(0.82, ckd | diab, 0.12)
    m_fbs = present(0.82, diab, 0.12)
    m_a1c = present(0.25, diab, 0.35)
    m_lft = present(0.55, vir | fatty, 0.15)
    m_tsh = present(0.40, hypo | hyper, 0.40)
    m_ft4 = m_tsh & (rng.random(n) < 0.35 + 0.4 * (hypo | hyper))
    m_core = rng.random(n) < 0.985   # occasional missing red-cell indices
    for cols, m in [(["neut_pct", "lymph_pct"], m_diff), (["total_chol", "ldl", "hdl", "tg"], m_lip),
                    (["creatinine", "urea"], m_ren), (["fbs"], m_fbs), (["hba1c"], m_a1c),
                    (["alt", "ast"], m_lft), (["tsh"], m_tsh), (["ft4"], m_ft4),
                    (["mcv", "mch", "mchc", "rbc", "hct"], m_core)]:
        for c in cols:
            df.loc[~m, c] = np.nan

    # ---- ground-truth labels with annotation noise --------------------------------------------
    y = pd.DataFrame(dict(anemia=anem, ckd=ckd, diabetes=diab, hypercholesterolemia=chol,
                          hypothyroidism=hypo, hyperthyroidism=hyper, infection=inf)).astype(int)
    drop = (rng.random(y.shape) < 0.03) & (y.values == 1)
    add = (rng.random(y.shape) < 0.003) & (y.values == 0)
    y = pd.DataFrame(np.where(drop, 0, np.where(add, 1, y.values)), columns=y.columns)

    out = pd.concat([df, y], axis=1)
    out["site"] = site
    return out


def generate(n_per_site: int = 5000, n_sites: int = 6, seed: int = 7) -> pd.DataFrame:
    """Sites 0..n_sites-2 are 'internal'; the last site is a shifted lab for external validation."""
    rng = np.random.default_rng(seed)
    frames = [generate_site(n_per_site, s, rng, shifted=(s == n_sites - 1)) for s in range(n_sites)]
    return pd.concat(frames, ignore_index=True)


def inject_ocr_errors(df: pd.DataFrame, rate: float = 0.04, seed: int = 0):
    """Simulate OCR decimal slips on one random present analyte per affected row.

    Returns (corrupted_df, factor) where factor is a DataFrame[float] holding the multiplier applied to each
    cell (1.0 = untouched). Decimal-point slips are x10 / x0.1; the milder x2 / x0.5 are digit-level misreads.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = [c for c in BOUNDS if c not in ("age",) and c in out.columns]
    factor = pd.DataFrame(1.0, index=out.index, columns=cols)
    for i in np.flatnonzero(rng.random(len(out)) < rate):
        idx = out.index[i]
        avail = [c for c in cols if pd.notna(out.at[idx, c])]
        if not avail:
            continue
        c = avail[rng.integers(len(avail))]
        f = float(rng.choice([10.0, 0.1, 10.0, 0.1, 2.0, 0.5]))
        out.at[idx, c] = out.at[idx, c] * f
        factor.at[idx, c] = f
    return out, factor
