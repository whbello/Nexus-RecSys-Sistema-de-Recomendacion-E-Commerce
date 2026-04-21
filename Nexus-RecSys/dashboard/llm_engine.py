"""
llm_engine.py — Motor LLM para nexus-recsys
============================================
Proveedor: Groq (LLaMA 3)
Modelo rápido:  llama3-8b-8192
Modelo calidad: llama3-70b-8192

Configurar: GROQ_API_KEY en .env o variable de entorno
Obtener key: https://console.groq.com/keys

Tres funcionalidades:
1. explain_recommendations()  — explica por qué se recomienda
2. parse_search_intent()      — convierte lenguaje natural en consulta
3. generate_search_response() — responde conversacionalmente
4. answer_metrics_question()  — responde preguntas técnicas del evaluador
"""

import json as json_module
import os
from typing import Optional


class LLMEngine:
    """
    Motor LLM centralizado para nexus-recsys.
    Se usa el proveedor Groq por su velocidad de inferencia.
    Si GROQ_API_KEY no está disponible, todos los métodos retornan un mensaje
    de fallback elegante para que el dashboard funcione igualmente.
    """

    SYSTEM_PROMPT = """Eres NEXUS AI, el asistente inteligente del sistema de
recomendación de productos nexus-recsys, desarrollado por Nexus Data Co.

Contexto del sistema:
- Modelo ganador: Mega-Ensemble NB15v2 (RP3+MB+TD · EASE^R · RP3+TD)
- NDCG@10 = 0.0431 (+50.8% sobre el baseline individual RP3+TD con 0.0286)
- Dataset: RetailRocket · 2.75M eventos · 1.4M usuarios · 235K ítems · 99.9994% sparsidad
- El deep learning (SASRec, NCF, BPR-MF) falló por sparsidad extrema del dataset
- RP3beta funciona porque es robusto con pocos datos por usuario (mediana = 1 ítem)
- El ensemble supera al mejor modelo individual porque cada componente
  ve el problema desde un ángulo distinto (correlación Spearman = 0.216)
- Pesos del ensemble: RP3+MB+TD (95.6%), EASE^R (2.1%), RP3+TD (2.3%)
- Split temporal: entrenamiento < 2015-08-22, test ≥ 2015-08-22
- Evaluación sobre 3000 usuarios warm seleccionados con seed=42

Tu personalidad:
- Experto pero accesible — explicás conceptos técnicos en lenguaje simple
- Honesto sobre las limitaciones del sistema
- Orientado al negocio — siempre conectás la técnica con el valor comercial
- Conciso — máximo 4 oraciones salvo que te pidan más detalle
- Respondés en español

Nunca inventes métricas ni resultados. Usá solo los números del contexto anterior."""

    # Preguntas frecuentes predefinidas para el Consultor de Métricas
    QUICK_QUESTIONS = [
        "¿Por qué el NDCG@10 es 0.0431 y no más alto?",
        "¿Cómo funciona el ensemble de 3 modelos?",
        "¿Por qué falló el deep learning (SASRec, NCF)?",
        "¿Qué es el cold-start y cómo lo manejan?",
        "¿Por qué usan split temporal y no aleatorio?",
        "¿Qué significa Temporal Decay en RP3beta?",
        "¿Cómo se eligió NDCG y no otra métrica?",
        "¿Qué haría el sistema con datos de stock y precio?",
    ]

    def __init__(self):
        """
        Inicializa el motor LLM.
        Lanza un ValueError si GROQ_API_KEY no está configurada.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY no encontrada. "
                "Configurala en el archivo .env o como variable de entorno."
            )
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError(
                "La librería 'groq' no está instalada. "
                "Ejecutá: pip install groq>=0.4.0"
            )
        # Leer modelos desde variables de entorno (con fallback a modelos actuales)
        self.fast_model    = os.environ.get("GROQ_FAST_MODEL",    "llama-3.1-8b-instant")    # velocidad
        self.quality_model = os.environ.get("GROQ_QUALITY_MODEL", "llama-3.3-70b-versatile")  # calidad

    def _call(
        self,
        prompt: str,
        use_quality: bool = False,
        max_tokens: int = 250,
        temperature: float = 0.7,
    ) -> str:
        """Llamada base al API de Groq con manejo de errores."""
        try:
            response = self.client.chat.completions.create(
                model=self.quality_model if use_quality else self.fast_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[NEXUS AI no disponible en este momento: {str(e)[:100]}]"

    # ── FUNCIÓN 1: EXPLICADOR DE RECOMENDACIONES ──────────────────────────────
    def explain_recommendations(
        self,
        user_id: int,
        user_history: list,
        recommendations: list,
        user_profile: dict,
    ) -> str:
        """
        Genera explicación en lenguaje natural de por qué se recomiendan
        estos productos a este usuario específico.

        Parámetros:
            user_history:    [{"name": str, "category": str, "event": str}]
            recommendations: [{"name": str, "category": str, "score": float}]
            user_profile:    {"n_interactions": int, "user_type": str, "top_categories": str}
        """
        history_str = "\n".join([
            f"  - {h.get('name', 'Producto desconocido')} "
            f"({h.get('category', '?')}) · {h.get('event', 'interacción')}"
            for h in user_history[-5:]
        ]) or "  Sin historial disponible (usuario nuevo)"

        recs_str = "\n".join([
            f"  {i+1}. {r.get('name', 'Producto')} ({r.get('category', '?')}) "
            f"· score: {r.get('score', 0):.3f}"
            for i, r in enumerate(recommendations[:5])
        ])

        prompt = f"""Un usuario acaba de recibir estas recomendaciones de nuestro sistema.

PERFIL DEL USUARIO (ID: {user_id}):
- Tipo: {user_profile.get('user_type', 'usuario activo')}
- Total de interacciones: {user_profile.get('n_interactions', 'N/A')}
- Categorías favoritas: {user_profile.get('top_categories', 'variadas')}

HISTORIAL RECIENTE:
{history_str}

PRODUCTOS RECOMENDADOS:
{recs_str}

Explicá en 3-4 oraciones naturales y amigables por qué el sistema eligió estos productos
para este usuario específico. Conectá el historial con las recomendaciones.
No uses tecnicismos. Hablá directamente al usuario ("te recomendamos...")."""

        return self._call(prompt, use_quality=False, max_tokens=200, temperature=0.8)

    # ── FUNCIÓN 2: BÚSQUEDA CONVERSACIONAL ───────────────────────────────────
    def parse_search_intent(self, user_message: str) -> dict:
        """
        Interpreta una búsqueda en lenguaje natural.
        Retorna un dict con: category, context, keywords, price_range, summary.
        """
        prompt = f"""Un usuario busca productos con esta descripción:
"{user_message}"

Analizá la intención y respondé SOLO con un JSON válido con esta estructura exacta:
{{
    "category": "categoría principal inferida (Electrónica/Ropa y Moda/Hogar y Decoración/Deportes y Fitness/Libros y Educación/Belleza y Cuidado Personal/Alimentos y Bebidas/general)",
    "context": "para regalo/uso personal/trabajo/deporte/etc.",
    "keywords": ["palabra1", "palabra2", "palabra3"],
    "price_range": "bajo/medio/alto/cualquiera",
    "summary": "resumen de 1 oración de lo que busca el usuario"
}}

Solo el JSON, sin texto adicional."""

        raw = self._call(prompt, use_quality=False, max_tokens=150, temperature=0.3)
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            return json_module.loads(raw[start:end])
        except Exception:
            return {
                "category":    "general",
                "context":     "uso personal",
                "keywords":    user_message.split()[:3],
                "price_range": "cualquiera",
                "summary":     user_message,
            }

    def generate_search_response(
        self,
        user_message: str,
        intent: dict,
        found_products: list,
    ) -> str:
        """
        Genera respuesta conversacional después de interpretar la búsqueda.
        """
        products_str = "\n".join([
            f"  - {p.get('name', '?')} ({p.get('category', '?')})"
            for p in found_products[:4]
        ]) or "  No encontré productos exactos, pero tengo alternativas."

        prompt = f"""El usuario escribió: "{user_message}"

Interpretaste que busca: {intent.get('summary', '')}
Categoría: {intent.get('category', '')}
Contexto: {intent.get('context', '')}

Los productos que encontró el sistema:
{products_str}

Respondé de forma conversacional y amigable en 2-3 oraciones:
1. Confirmá que entendiste lo que busca
2. Presentá las recomendaciones de forma natural
3. Preguntá si quiere más opciones o algo más específico"""

        return self._call(prompt, use_quality=False, max_tokens=150, temperature=0.8)

    # ── FUNCIÓN 3: CONSULTOR DE MÉTRICAS ─────────────────────────────────────
    def answer_metrics_question(
        self,
        question: str,
        additional_context: Optional[dict] = None,
    ) -> str:
        """
        Responde preguntas técnicas del evaluador sobre el modelo.
        Usa el modelo de mayor calidad para respuestas más precisas.
        """
        ctx = ""
        if additional_context:
            ctx = f"\nContexto adicional: {json_module.dumps(additional_context, ensure_ascii=False)}"

        prompt = f"""Un evaluador académico pregunta sobre el sistema de recomendación:

"{question}"
{ctx}

Respondé de forma técnica pero clara en máximo 4 oraciones.
Sé específico con los números y decisiones del proyecto.
Si la pregunta es sobre limitaciones, sé honesto y mencioná las oportunidades de mejora.
Si es sobre trade-offs, explicá qué se ganó y qué se perdió con cada decisión."""

        return self._call(
            prompt, use_quality=True, max_tokens=280, temperature=0.3
        )


def try_load_engine() -> tuple:
    """
    Intenta cargar el LLMEngine.
    Retorna (engine, None) si tiene éxito, (None, error_message) si falla.
    Útil para el dashboard: si no hay API key, sigue funcionando.
    """
    try:
        engine = LLMEngine()
        return engine, None
    except (ValueError, ImportError) as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error inesperado al inicializar LLM: {str(e)[:100]}"
