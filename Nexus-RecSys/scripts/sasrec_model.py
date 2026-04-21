"""
SASRec — Self-Attentive Sequential Recommendation.
Kang & McAuley, ICDM 2018.

Implementación PyTorch pura, corregida respecto a SASRec-lite de NB09.
Correcciones principales:
  - Embeddings posicionales APRENDIDOS (no sinusoidales fijos)
  - Multi-head attention completo con padding mask + causal mask
  - Feed-forward network con dos capas lineales y GELU
  - Residual connections + Layer Norm Pre-LN en cada sub-capa
  - Índice 0 reservado para padding (n_items+1 embeddings totales)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointWiseFeedForward(nn.Module):
    """
    Red feed-forward posición a posición.
    Dos capas lineales con activación GELU y dropout.
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Un bloque transformer con Pre-LN (layer norm antes de cada sub-capa).
    Sub-capa 1: Multi-Head Self-Attention con causal mask + padding mask.
    Sub-capa 2: Feed-Forward posición a posición.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-8)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-8)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn     = PointWiseFeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask, attn_mask):
        """
        x:                 [batch, seq_len, d_model]
        key_padding_mask:  [batch, seq_len]  True en posiciones de padding
        attn_mask:         [seq_len, seq_len] máscara causal triangular superior
        """
        # Sub-capa 1: Self-Attention con residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # Sub-capa 2: FFN con residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation.
    Kang & McAuley, ICDM 2018.

    Índice 0 reservado para padding → total embeddings = n_items + 1.
    Los ítems del catálogo se indexan desde 1 hasta n_items.
    """
    def __init__(
        self,
        n_items:  int,
        maxlen:   int   = 20,
        d_model:  int   = 64,
        n_heads:  int   = 2,
        n_layers: int   = 2,
        dropout:  float = 0.5,
        device:   str   = "cpu",
    ):
        super().__init__()
        self.n_items  = n_items
        self.maxlen   = maxlen
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.device   = device

        # Embedding de ítems: índice 0 = padding (relleno de secuencias cortas)
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)

        # Embedding posicional APRENDIDO (no sinusoidal)
        self.pos_emb  = nn.Embedding(maxlen, d_model)

        self.emb_dropout = nn.Dropout(dropout)
        self.norm_out    = nn.LayerNorm(d_model, eps=1e-8)

        # Capas transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        """Inicialización normal con std=0.02 (estilo BERT)."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Máscara causal triangular superior para atención autoregresiva.
        True = ignorar esa posición futura.
        Shape: [seq_len, seq_len]
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
            diagonal=1,
        )
        return mask

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante.

        seqs: [batch, seq_len]  — índices de ítems, 0 = padding

        Retorna logits: [batch, seq_len, n_items+1]
        """
        batch, seq_len = seqs.shape

        # Posiciones: [0, 1, ..., seq_len-1] repetidas para el batch
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch, -1)

        # Suma de embeddings de ítems y posicionales
        x = self.emb_dropout(self.item_emb(seqs) + self.pos_emb(positions))

        # Máscara de padding: True donde el ítem es 0 (padding)
        key_padding_mask = (seqs == 0)  # [batch, seq_len]

        # Máscara causal: no ver posiciones futuras
        attn_mask = self._build_causal_mask(seq_len)  # [seq_len, seq_len]

        # Bloques transformer
        for block in self.blocks:
            x = block(x, key_padding_mask, attn_mask)

        # Normalización final de salida
        x = self.norm_out(x)

        # Proyección sobre todos los ítems del catálogo (incluyendo padding=0)
        logits = x @ self.item_emb.weight.T  # [batch, seq_len, n_items+1]
        return logits

    def get_hidden(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Retorna representaciones ocultas sin la proyección final al catálogo.
        seqs: [batch, seq_len]
        Retorna: [batch, seq_len, d_model]
        Nota: más eficiente que forward() cuando solo se necesitan un subconjunto
        de scores (training con BCE selectivo).
        """
        positions = torch.arange(self.maxlen, device=seqs.device).unsqueeze(0)
        x = self.emb_dropout(self.item_emb(seqs) + self.pos_emb(positions))
        key_padding_mask = (seqs == 0)
        attn_mask = self._build_causal_mask(self.maxlen)
        for block in self.blocks:
            x = block(x, key_padding_mask, attn_mask)
        return self.norm_out(x)  # [batch, seq_len, d_model]

    @torch.no_grad()
    def predict(
        self,
        seq: torch.Tensor,
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        Genera ranking top-K para una única secuencia.

        seq: [1, seq_len] con la secuencia del usuario

        Retorna índices de los top-K ítems (1-indexed).
        """
        self.eval()
        logits = self.forward(seq)          # [1, seq_len, n_items+1]
        last   = logits[0, -1, 1:]         # [n_items]  ignorar índice 0
        top_ids = torch.argsort(last, descending=True)[:top_k] + 1  # índices 1-based
        return top_ids

    @torch.no_grad()
    def get_all_scores(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Retorna scores sobre todos los ítems del catálogo (1-indexed).
        Proyecta solo la última posición oculta sobre el catálogo completo,
        evitando el costoso [batch, seq_len, n_items+1].
        seq: [1, seq_len]
        Retorna: [n_items] scores
        """
        self.eval()
        hidden   = self.get_hidden(seq)          # [1, seq_len, d_model]
        last_h   = hidden[0, -1, :]              # [d_model]
        scores   = last_h @ self.item_emb.weight.T  # [n_items+1]
        return scores[1:]                        # ignorar padding_idx=0  → [n_items]


class SASRecTrainer:
    """
    Clase de entrenamiento de SASRec con early stopping.
    """
    def __init__(self, model: SASRec, config: dict):
        self.model  = model
        self.config = config
        self.device = config.get("device", "cpu")

        # Optimizador: Adam con L2 opcional
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("l2_emb", 0.0),
        )

        # Historial de entrenamiento
        self.history = {
            "epoch":         [],
            "train_loss":    [],
            "val_ndcg10":    [],
            "val_hr10":      [],
            "tiempo_epoch_s": [],
        }
        self.best_val_ndcg  = -np.inf
        self.best_epoch     = 0
        self.patience_count = 0
        self.best_state     = None

    @staticmethod
    def bce_loss_sasrec(
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary Cross Entropy sobre pares (positivo, negativo).

        pos_logits: [batch, seq_len]  scores del ítem siguiente real
        neg_logits: [batch, seq_len]  scores de ítem negativo sampleado
        mask:       [batch, seq_len]  True en posiciones válidas (no padding)

        Devuelve pérdida promedio sobre posiciones válidas.
        """
        # Pérdida BCE: − [log σ(pos) + log σ(−neg)]
        loss_pos = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction="none"
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction="none"
        )
        loss = (loss_pos + loss_neg) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    def sample_negatives(
        self,
        seqs: torch.Tensor,
        n_items: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """
        Sampleado uniforme de ítems negativos.
        Para cada posición (batch, t): samplear un ítem que NO esté
        en el historial del usuario (seqs[batch]).

        seqs: [batch, seq_len]
        Retorna: [batch, seq_len] negs
        """
        batch_size, seq_len = seqs.shape
        negs = np.zeros((batch_size, seq_len), dtype=np.int64)
        seqs_np = seqs.cpu().numpy()
        for i in range(batch_size):
            user_items = set(seqs_np[i].tolist()) - {0}
            for t in range(seq_len):
                neg = rng.integers(1, n_items + 1)
                while neg in user_items:
                    neg = rng.integers(1, n_items + 1)
                negs[i, t] = neg
        return torch.from_numpy(negs).to(seqs.device)

    def evaluate_lou(
        self,
        sequences_val: list,
        k_list: list = [5, 10],
        max_users: int = 1000,
    ) -> dict:
        """
        Evaluación leave-one-out sobre el validation set.
        Para cada usuario: ranking del ítem target contra TODOS los ítems.

        sequences_val: lista de (seq_train, target_item)
        k_list:        umbrales K para HR@K y NDCG@K
        max_users:     límite de usuarios para acelerar evaluación en CPU

        Retorna dict con HR@K y NDCG@K para cada K.
        """
        import copy
        self.model.eval()

        # Limitar número de usuarios para acelerar evaluación en CPU
        data = sequences_val[:max_users]
        n = len(data)

        hits  = {k: 0 for k in k_list}
        ndcgs = {k: 0.0 for k in k_list}

        with torch.no_grad():
            for seq_train, target in data:
                # Preparar secuencia: truncar/padear a maxlen
                seq_arr = np.array(seq_train, dtype=np.int64)
                maxlen  = self.model.maxlen
                if len(seq_arr) >= maxlen:
                    seq_arr = seq_arr[-maxlen:]
                else:
                    seq_arr = np.concatenate([np.zeros(maxlen - len(seq_arr), dtype=np.int64), seq_arr])

                seq_t = torch.LongTensor(seq_arr).unsqueeze(0).to(
                    next(self.model.parameters()).device
                )
                scores = self.model.get_all_scores(seq_t).cpu().numpy()  # [n_items]

                # Posición (rank) del ítem target (1-indexed)
                rank = int((scores > scores[target - 1]).sum()) + 1  # target es 1-indexed

                for k in k_list:
                    if rank <= k:
                        hits[k]  += 1
                        ndcgs[k] += 1.0 / np.log2(rank + 1)

        metrics = {}
        for k in k_list:
            metrics[f"HR@{k}"]   = hits[k]  / n
            metrics[f"NDCG@{k}"] = ndcgs[k] / n
        return metrics

    def train(
        self,
        train_data: list,
        val_data:   list,
        verbose:    bool = True,
    ) -> dict:
        """
        Loop de entrenamiento completo con early stopping.

        train_data: lista de (seq_input, seq_output) donde ambas son
                    listas de índices 1-indexed de ítems.
        val_data:   lista de (seq_train, target_item) para evaluación LOU.

        Retorna historial de entrenamiento.
        """
        import time
        import copy

        config     = self.config
        max_epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 128)
        eval_every = config.get("eval_every", 5)
        patience   = config.get("patience", 10)
        maxlen     = self.model.maxlen
        n_items    = self.model.n_items
        device     = self.device
        rng        = np.random.default_rng(config.get("random_state", 42))

        if verbose:
            print(f"  Iniciando entrenamiento SASRec: {len(train_data):,} usuarios "
                  f"| {max_epochs} epochs | batch={batch_size}")
            print(f"  {'Epoch':>6}  {'Loss':>8}  {'ValNDCG@10':>12}  {'ValHR@10':>10}  {'Tiempo':>8}")
            print("  " + "-" * 55)

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            self.model.train()

            # Mezclar datos de train con seed controlada
            indices = rng.permutation(len(train_data))
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start: start + batch_size]
                batch     = [train_data[i] for i in batch_idx]

                # Construir tensores de entrada y salida
                seqs_in  = np.zeros((len(batch), maxlen), dtype=np.int64)
                seqs_out = np.zeros((len(batch), maxlen), dtype=np.int64)
                for b, (seq_in, seq_out) in enumerate(batch):
                    # Tomar los últimos maxlen elementos
                    sl = min(len(seq_in), maxlen)
                    seqs_in [b, -sl:] = seq_in [-sl:]
                    sl2 = min(len(seq_out), maxlen)
                    seqs_out[b, -sl2:] = seq_out[-sl2:]

                seqs_in_t  = torch.LongTensor(seqs_in ).to(device)
                seqs_out_t = torch.LongTensor(seqs_out).to(device)

                # Máscara válida: posiciones donde hay ítem de salida real
                mask = (seqs_out_t != 0).float()

                # ── Scoring eficiente con dot-product selectivo ──────────────
                # Avoids computing full [batch, seq_len, n_items+1] projection.
                # Compute hidden states: [batch, seq_len, d_model]
                hidden = self.model.get_hidden(seqs_in_t)

                # Embeddings de ítems positivos: [batch, seq_len, d_model]
                pos_embs = self.model.item_emb(seqs_out_t.clamp(min=0))
                pos_scores = (hidden * pos_embs).sum(-1)  # [batch, seq_len]

                # Samplear negativos y calcular sus scores por dot-product
                neg_items  = self.sample_negatives(seqs_out_t, n_items, rng)
                neg_embs   = self.model.item_emb(neg_items)
                neg_scores = (hidden * neg_embs).sum(-1)  # [batch, seq_len]

                # Pérdida BCE
                loss = self.bce_loss_sasrec(pos_scores, neg_scores, mask)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time.time() - t0

            # Evaluar en validation cada eval_every epochs
            if epoch % eval_every == 0 or epoch == 1:
                val_metrics = self.evaluate_lou(
                    val_data,
                    k_list=[5, 10],
                    max_users=config.get("val_max_users", 500),
                )
                val_ndcg10 = val_metrics.get("NDCG@10", 0.0)
                val_hr10   = val_metrics.get("HR@10",   0.0)

                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(avg_loss)
                self.history["val_ndcg10"].append(val_ndcg10)
                self.history["val_hr10"].append(val_hr10)
                self.history["tiempo_epoch_s"].append(elapsed)

                if verbose:
                    print(f"  {epoch:>6}  {avg_loss:>8.4f}  {val_ndcg10:>12.5f}  "
                          f"{val_hr10:>10.4f}  {elapsed:>6.1f}s")

                # Guardar mejor checkpoint
                if val_ndcg10 > self.best_val_ndcg:
                    self.best_val_ndcg  = val_ndcg10
                    self.best_epoch     = epoch
                    self.patience_count = 0
                    self.best_state     = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_count += 1
                    if self.patience_count >= patience:
                        if verbose:
                            print(f"\n  Early stopping en epoch {epoch}. "
                                  f"Mejor epoch: {self.best_epoch} "
                                  f"(NDCG@10={self.best_val_ndcg:.5f})")
                        break
            else:
                # Solo registrar el loss sin evaluar
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(avg_loss)
                self.history["val_ndcg10"].append(None)
                self.history["val_hr10"].append(None)
                self.history["tiempo_epoch_s"].append(elapsed)

        # Restaurar mejor checkpoint
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        if verbose:
            print(f"\n  Entrenamiento finalizado. "
                  f"Mejor NDCG@10 val={self.best_val_ndcg:.5f} "
                  f"en epoch {self.best_epoch}")

        return self.history
