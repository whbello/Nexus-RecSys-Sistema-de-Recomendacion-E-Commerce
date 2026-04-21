"""
multivae_model.py
=================
Implementacion PyTorch pura de Mult-VAE^PR (Multinomial Variational Autoencoder
para Recomendacion con Partial Regularization).

Referencia:
  Liang, D., Krishnan, R.G., Hoffman, M.D., & Jebara, T. (2018).
  Variational Autoencoders for Collaborative Filtering. WWW 2018.

Arquitectura:
  Encoder: x -> [h1] -> [h2] -> mu / log_var
  Decoder: z -> [h2] -> [h1] -> logits  (tamaño catálogo)
  Loss:    L = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
           con KL annealing: beta sube de 0 a beta_max durante warmup_epochs

Notas de ingeniería:
  - Entrada: vector sparse binarizado del historial del usuario
  - Likelihood: multinomial (log-softmax + dot product)
  - Dropout de entrada (corruption): regulariza el encoder
  - El modelo trabaja sobre el subconjunto top_K_items (los más populares)
    para mantener n_items manejable en CPU
"""

import math
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UserHistoryDataset(Dataset):
    """
    Dataset de vectores de historial de usuario (fila de R).

    Parámetros
    ----------
    R : scipy.sparse.csr_matrix  (n_users, n_items)
        Matriz de interacciones (binarizada / ponderada), subconjunto de top_items.
    user_indices : np.ndarray
        Índices de filas (usuarios) que forman este split.
    binary : bool
        Si True, binariza los valores (cualquier valor > 0 → 1.0).
        La multinomial likelihood es más estable con inputs binarizados.
    """

    def __init__(
        self,
        R: sp.csr_matrix,
        user_indices: np.ndarray,
        binary: bool = True,
    ):
        self.R = R
        self.user_indices = user_indices
        self.binary = binary

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ui = self.user_indices[idx]
        row = self.R.getrow(ui)
        x = np.asarray(row.todense(), dtype=np.float32).ravel()
        if self.binary:
            x = (x > 0).astype(np.float32)
        return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class MultiVAE(nn.Module):
    """
    Mult-VAE^PR: Variational Autoencoder con Multinomial Likelihood.

    Parámetros
    ----------
    n_items : int
        Tamaño del vocabulario de ítems (después de filtrar por top_K).
    enc_dims : List[int]
        Dimensiones ocultas del encoder (sin incluir n_items ni latent_dim).
        Ej: [600, 200] → Linear(n_items, 600) → Linear(600, 200) → Linear(200, latent_dim*2)
    latent_dim : int
        Dimensión del espacio latente z.
    dropout_rate : float
        Dropout aplicado al input x (corruption noise).
    """

    def __init__(
        self,
        n_items: int,
        enc_dims: List[int] = [600, 200],
        latent_dim: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.n_items = n_items
        self.latent_dim = latent_dim
        self.input_dropout = nn.Dropout(p=dropout_rate)

        # ---------- Encoder ----------
        enc_layers = []
        in_dim = n_items
        for h in enc_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.Tanh())
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        # Proyección a mu y log_var (concatenados en un vector de 2*latent_dim)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # ---------- Decoder ----------
        dec_dims = enc_dims[::-1]  # Espejo del encoder
        dec_layers = []
        in_dim = latent_dim
        for h in dec_dims:
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.Tanh())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_items))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization para todas las capas lineales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, n_items)  float, binarizado o ponderado
        Returns: mu (batch, latent_dim), log_var (batch, latent_dim)
        """
        # Normalización L2 del input (estabiliza el encoder con inputs de distinta densidad)
        x_norm = F.normalize(x, p=2, dim=1)
        x_drop = self.input_dropout(x_norm)
        h = self.encoder(x_drop)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std, eps ~ N(0,I)
        En evaluación (self.training=False) devuelve directamente mu.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch, latent_dim)
        Returns: logits (batch, n_items)  — sin softmax (usamos log_softmax en la loss)
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: logits, mu, log_var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        logits = self.decode(z)
        return logits, mu, log_var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def multivae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    anneal_cap: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Multinomial ELBO:
      L = -E_q[log p(x|z)] + beta * KL(q(z|x) || p(z))

    - Reconstruction: multinomial log-likelihood = sum_i x_i * log_softmax(logits)_i
      (equivalente a cross-entropy pesada donde el target no es one-hot)
    - KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Parámetros
    ----------
    logits : (batch, n_items)
    x      : (batch, n_items)  input binarizado (pseudo-counts)
    mu     : (batch, latent_dim)
    log_var: (batch, latent_dim)
    beta   : peso del KL (annealed)
    anneal_cap : valor máximo del annealing (normalmente 1.0)

    Returns: loss (escalar), recon_mean, kl_mean
    """
    # Multinomial NLL (equivalente a cross-entropy con target x normalizado)
    log_softmax = F.log_softmax(logits, dim=1)
    recon = -torch.sum(x * log_softmax, dim=1).mean()

    # KL divergence
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    beta_eff = min(beta, anneal_cap)
    loss = recon + beta_eff * kl

    return loss, recon.item(), kl.item()


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_multivae(
    R: sp.csr_matrix,
    train_user_idx: np.ndarray,
    n_items: int,
    enc_dims: List[int] = [600, 200],
    latent_dim: int = 64,
    dropout_rate: float = 0.5,
    lr: float = 1e-3,
    l2_reg: float = 1e-5,
    n_epochs: int = 50,
    batch_size: int = 512,
    beta_max: float = 1.0,
    warmup_epochs: int = 10,
    total_anneal_steps: int = 200_000,
    seed: int = 42,
    verbose: bool = True,
    device: Optional[str] = None,
) -> Tuple["MultiVAE", dict]:
    """
    Entrena MultiVAE sobre el subconjunto de train_user_idx.

    Estrategia de KL annealing:
      - Durante los primeros `warmup_epochs` epochs, beta sube linealmente de 0 a beta_max
        (basado en número de steps, no de epochs, para granularidad fina)

    Parámetros
    ----------
    R : csr_matrix (n_users_total, n_items)
        Solo se usarán las filas indicadas por train_user_idx.
    train_user_idx : array de índices de usuarios para entrenamiento
    n_items : tamaño del vocabulario (debe coincidir con R.shape[1])
    enc_dims : dimensiones ocultas del encoder
    latent_dim : dimensión Z
    dropout_rate : dropout en el input
    lr : learning rate para Adam
    l2_reg : weight decay (L2 regularización sobre parámetros)
    n_epochs : número de epochs
    batch_size : tamaño de mini-batch
    beta_max : valor máximo del coeficiente KL
    warmup_epochs : epochs para llegar de beta=0 a beta=beta_max
    total_anneal_steps : alternativa de annealing basada en steps
    seed : semilla
    verbose : si True imprime progreso por epoch
    device : 'cuda', 'cpu' o None (auto-detect)

    Returns: modelo entrenado, historial de loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiVAE(
        n_items=n_items,
        enc_dims=enc_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=l2_reg
    )

    dataset = UserHistoryDataset(R, train_user_idx, binary=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    n_steps = len(loader)
    anneal_per_step = beta_max / max(total_anneal_steps, 1)
    beta = 0.0
    global_step = 0

    history = {"loss": [], "recon": [], "kl": [], "beta": []}

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0

        for batch in loader:
            x = batch.to(device)
            logits, mu, log_var = model(x)
            loss, recon, kl = multivae_loss(logits, x, mu, log_var, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping para estabilidad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # KL annealing: incremento lineal por step
            beta = min(beta_max, beta + anneal_per_step)
            global_step += 1

            epoch_loss += loss.item()
            epoch_recon += recon
            epoch_kl += kl

        nb = len(loader)
        avg_loss = epoch_loss / nb
        avg_recon = epoch_recon / nb
        avg_kl = epoch_kl / nb

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        history["beta"].append(beta)

        if verbose:
            elapsed = time.time() - t_start
            print(
                f"  MultiVAE ep {epoch:3d}/{n_epochs}  "
                f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  "
                f"kl={avg_kl:.4f}  beta={beta:.4f}  [{elapsed:.1f}s]"
            )

    return model, history


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def build_scorer(
    model: "MultiVAE",
    R: sp.csr_matrix,
    user2idx: dict,
    top_items_global: list,
    device: Optional[str] = None,
    batch_inference: bool = True,
) -> callable:
    """
    Devuelve una función get_multivae(uid, n) compatible con evaluate() de NB09.

    La función:
      1. Recupera la fila sparse del usuario de R
      2. Binariza y normaliza (L2)
      3. Pasa por el encoder → mu (no muestrea en inferencia)
      4. Pasa por el decoder → logits
      5. Máscara de ítems ya vistos
      6. Devuelve top-n ítems (ids globales)

    Parámetros
    ----------
    model : MultiVAE entrenado
    R : csr_matrix (n_users_total, n_items_local)   — subconjunto top_items
    user2idx : dict uid_global → row_idx en R
    top_items_global : lista de item_ids globales (len = n_items_local),
                       ordenada igual que las columnas de R
    device : device del modelo
    batch_inference : no usado aquí (reservado para scoring en batch)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    def get_multivae(uid: int, n: int) -> list:
        if uid not in user2idx:
            return []
        ui = user2idx[uid]

        # Construcción del vector de input
        row = R.getrow(ui)
        x = np.asarray(row.todense(), dtype=np.float32).ravel()
        x_bin = (x > 0).astype(np.float32)

        x_tensor = torch.from_numpy(x_bin).unsqueeze(0).to(device)  # (1, n_items)

        with torch.no_grad():
            logits, _, _ = model(x_tensor)  # (1, n_items)
        scores = logits.squeeze(0).cpu().numpy()  # (n_items,)

        # Mascarar ítems ya vistos (interaciones en train)
        scores[x_bin > 0] = -np.inf

        # Top-n
        n_safe = min(n, len(scores))
        top_local = np.argpartition(scores, -n_safe)[-n_safe:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return [top_items_global[i] for i in top_local]

    return get_multivae


# ---------------------------------------------------------------------------
# Pipeline completo (helper de alto nivel para NB10)
# ---------------------------------------------------------------------------

def run_multivae_pipeline(
    R_full: sp.csr_matrix,
    user2idx: dict,
    idx2item: dict,
    item_pop: np.ndarray,
    eval_users: list,
    test_items_by_user: dict,
    test_tx_by_user: dict,
    item_pop_dict: dict,
    n_total_train: float,
    n_items_global: int,
    baseline_conv: float,
    # Architectural hyperparams
    top_k_items: int = 20_000,
    enc_dims: List[int] = [600, 200],
    latent_dim: int = 64,
    dropout_rate: float = 0.5,
    # Training hyperparams
    lr: float = 1e-3,
    l2_reg: float = 1e-5,
    n_epochs: int = 50,
    batch_size: int = 512,
    beta_max: float = 0.3,
    total_anneal_steps: int = 200_000,
    # Subset usuarios para escalabilidad CPU
    max_train_users: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
    # evaluate() K values
    ks: List[int] = [5, 10, 20],
) -> Tuple["MultiVAE", dict, dict]:
    """
    Pipeline end-to-end: preprocesamiento → entrenamiento → evaluación.

    Devuelve: modelo, historial de entrenamiento, métricas de evaluación.

    Nota sobre max_train_users:
      El encoder aprende representaciones de usuarios por su historial.
      Usuarios con historial muy corto (1 ítem) aportan poco al entrenamiento
      pero ralentizan el processo. Se puede subsetear a usuarios con >= 2 items
      para un entrenamiento más eficiente en CPU.
    """
    t0 = time.time()

    # 1. Selección de top-K ítems (mismo criterio que EASE^R en NB09)
    top_idx = np.argpartition(item_pop, -top_k_items)[-top_k_items:]
    top_idx = top_idx[np.argsort(item_pop[top_idx])[::-1]]
    n_items = len(top_idx)

    top_items_global = [idx2item[i] for i in top_idx]

    # Submatriz sparse (n_users_total, n_items_local)
    R_sub = R_full[:, top_idx].astype(np.float32).tocsr()
    if verbose:
        print(f"[MultiVAE] R_sub: {R_sub.shape}  nnz={R_sub.nnz:,}  [{time.time()-t0:.1f}s]")

    # 2. Selección de usuarios para entrenamiento
    # Usar todos los usuarios que aparecen en las filas (tienen al menos 1 ítem)
    row_sums = np.asarray(R_sub.sum(axis=1)).ravel()
    active_idx = np.where(row_sums > 0)[0]

    if max_train_users is not None and len(active_idx) > max_train_users:
        # Priorizar usuarios con más ítems (más señal para el VAE)
        rng = np.random.default_rng(seed)
        # Usuarios con >= 2 ítems primero
        rich_idx = active_idx[row_sums[active_idx] >= 2]
        poor_idx = active_idx[row_sums[active_idx] < 2]

        if len(rich_idx) >= max_train_users:
            train_user_idx = rng.choice(rich_idx, size=max_train_users, replace=False)
        else:
            n_poor = min(max_train_users - len(rich_idx), len(poor_idx))
            extra = rng.choice(poor_idx, size=n_poor, replace=False)
            train_user_idx = np.concatenate([rich_idx, extra])
    else:
        train_user_idx = active_idx

    if verbose:
        print(f"[MultiVAE] Train users: {len(train_user_idx):,}  [{time.time()-t0:.1f}s]")

    # 3. Entrenamiento
    model, history = train_multivae(
        R=R_sub,
        train_user_idx=train_user_idx,
        n_items=n_items,
        enc_dims=enc_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        lr=lr,
        l2_reg=l2_reg,
        n_epochs=n_epochs,
        batch_size=batch_size,
        beta_max=beta_max,
        total_anneal_steps=total_anneal_steps,
        seed=seed,
        verbose=verbose,
    )

    # 4. Función de scoring compatible con evaluate()
    get_fn = build_scorer(model, R_sub, user2idx, top_items_global)

    # 5. Evaluación
    if verbose:
        print(f"[MultiVAE] Evaluando {len(eval_users):,} usuarios ...")

    t_eval = time.time()
    metrics = _evaluate_compat(
        get_fn,
        eval_users,
        test_items_by_user,
        test_tx_by_user,
        item_pop_dict,
        n_total_train,
        n_items_global,
        baseline_conv,
        ks=ks,
    )

    if verbose:
        print(
            f"[MultiVAE] eval={time.time()-t_eval:.1f}s  "
            f"NDCG@10={metrics.get('NDCG@10', 0):.4f}  "
            f"Coverage={metrics.get('Coverage', 0):.4f}  "
            f"total={time.time()-t0:.1f}s"
        )

    return model, history, metrics


def _evaluate_compat(
    get_fn,
    eval_users,
    test_items_by_user,
    test_tx_by_user,
    item_pop_dict,
    n_total_train,
    n_items_global,
    baseline_conv,
    ks,
):
    """
    Replica exacta de evaluate() de NB09 para compatibilidad total.
    Se incluye aquí para que multivae_model.py sea auto-contenido.
    """
    def _ndcg(r, rel, k):
        d = sum(1.0 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in rel)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
        return d / ideal if ideal else 0.0

    def _prec(r, rel, k): return len(set(r[:k]) & rel) / k if k else 0.0
    def _rec(r, rel, k):  return len(set(r[:k]) & rel) / len(rel) if rel else 0.0
    def _ap(r, rel, k):
        if not rel: return 0.0
        s, h = 0.0, 0
        for i, x in enumerate(r[:k]):
            if x in rel:
                h += 1; s += h / (i + 1)
        return s / min(len(rel), k)
    def _rev(r, tx, k): return len(set(r[:k]) & tx) / k if k else 0.0
    def _ctr(r, ts, k): return len(set(r[:k]) & ts) / k if k else 0.0
    def _nov(flat, pd, nt):
        return float(np.mean([-math.log2(pd.get(x, 1) / nt + 1e-10) for x in flat])) if flat else 0.0

    acc = {k: {m: [] for m in ["p", "r", "n", "m", "r2", "c"]} for k in ks}
    seen = set()
    ne = 0

    for uid in eval_users:
        ti = test_items_by_user.get(uid, set())
        if not ti:
            continue
        tx = test_tx_by_user.get(uid, set())
        mk = max(ks)
        try:
            recs = get_fn(uid, mk)
        except Exception:
            continue
        seen.update(recs)
        ne += 1
        for k in ks:
            acc[k]["p"].append(_prec(recs, ti, k))
            acc[k]["r"].append(_rec(recs, ti, k))
            acc[k]["n"].append(_ndcg(recs, ti, k))
            acc[k]["m"].append(_ap(recs, ti, k))
            acc[k]["r2"].append(_rev(recs, tx, k))
            acc[k]["c"].append(_ctr(recs, ti, k))

    out = {"n_eval": ne}
    for k in ks:
        if not acc[k]["p"]:
            continue
        out[f"NDCG@{k}"]      = float(np.mean(acc[k]["n"]))
        out[f"Precision@{k}"] = float(np.mean(acc[k]["p"]))
        out[f"Recall@{k}"]    = float(np.mean(acc[k]["r"]))
        out[f"MAP@{k}"]       = float(np.mean(acc[k]["m"]))
        out[f"Revenue@{k}"]   = float(np.mean(acc[k]["r2"]))
        out[f"CTR@{k}"]       = float(np.mean(acc[k]["c"]))
        rv = out[f"Revenue@{k}"]
        out[f"ConvLift@{k}"]  = rv / baseline_conv if baseline_conv else 0.0

    out["Coverage"] = len(seen) / n_items_global
    out["Novelty"]  = _nov(list(seen), item_pop_dict, n_total_train)
    return out
